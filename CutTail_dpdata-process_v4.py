import argparse
import dpdata
import glob
import logging
import multiprocessing
import os
import shutil
import sys

import numpy as np
import matplotlib.pyplot as plt

# ========================
# Configuration Variables
# ========================
INPUT_PATTERN = "OUTCAR_???"  # OUTCAR file naming pattern
STEPS_TO_REMOVE = 0           # Remove last N steps
SKIP_INTERVAL = 1             # Use every Nth step
OUTPUT_DIR = "processed_data"  # Temporary directory for processing
FINAL_DIR = "deepmd_data"      # Final merged directory
LOG_FILE = "process_log.txt"
dp_batch_size = 512
TRAIN_RATIO = 0.8             # Ratio for training data (validation = 1 - TRAIN_RATIO)
VISUALIZATION_FILE = "train_val_split.png"  # Output plot file
MAX_WORKERS = None
ENABLE_VISUALIZATION = True

LOGGER = logging.getLogger(__name__)
WORKER_SETTINGS = {}
# ========================

def _worker_init(settings):
    """Initializer for worker processes to receive configuration."""
    global WORKER_SETTINGS
    WORKER_SETTINGS = settings


def configure_logging(log_file, verbose=True):
    """Configure logging for both console and file outputs."""
    log_level = logging.INFO if verbose else logging.WARNING
    LOGGER.setLevel(log_level)

    # Remove existing handlers to avoid duplication when re-running main.
    for handler in LOGGER.handlers[:]:
        LOGGER.removeHandler(handler)
        handler.close()

    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)


def clean_outcar(file_in, file_out, steps_to_remove=None):
    """Clean OUTCAR file by removing the last ``steps_to_remove`` steps.

    A ``steps_to_remove`` value of 0 keeps all steps intact. The function
    returns ``True`` if the output file was successfully written.
    """
    if steps_to_remove is None:
        steps_to_remove = STEPS_TO_REMOVE
    with open(file_in, "r") as fin:
        lines = fin.readlines()

    position_indices = [
        i
        for i, line in enumerate(lines)
        if "POSITION" in line and "TOTAL-FORCE" in line
    ]

    if not position_indices:
        return False

    if steps_to_remove <= 0:
        with open(file_out, "w") as fout:
            fout.writelines(lines)
        return True

    if len(position_indices) <= steps_to_remove:
        return False

    # Index of the first "POSITION" line of the step(s) to be removed
    last_index_to_keep = position_indices[-steps_to_remove]
    with open(file_out, "w") as fout:
        fout.writelines(lines[:last_index_to_keep])
    return True

def process_single_outcar(outcar_file):
    """Process a single OUTCAR file to DeepMD format."""
    settings = WORKER_SETTINGS or {
        "output_dir": OUTPUT_DIR,
        "skip_interval": SKIP_INTERVAL,
        "batch_size": dp_batch_size,
        "steps_to_remove": STEPS_TO_REMOVE,
    }

    base_name = os.path.basename(outcar_file)
    key = int(base_name.split('_')[-1])

    output_dir = settings["output_dir"]
    npy_temp_dir = os.path.join(output_dir, f"set_temp_{key:03d}")
    LOGGER.info("    Processing %s...", base_name)
    outcar_clean = os.path.join(output_dir, f"{base_name}_clean")

    if os.path.exists(outcar_clean):
        os.remove(outcar_clean)

    if os.path.isdir(npy_temp_dir):
        shutil.rmtree(npy_temp_dir, ignore_errors=True)

    success = clean_outcar(outcar_file, outcar_clean, steps_to_remove=settings["steps_to_remove"])
    if not success:
        LOGGER.warning("Failed to clean %s. Skipping...", base_name)
        return None

    try:
        dsys = dpdata.LabeledSystem(outcar_clean, fmt="vasp/outcar")
        skip_interval = settings["skip_interval"]
        if skip_interval > 1:
            indices = range(0, dsys.get_nframes(), skip_interval)
            dsys = dsys.sub_system(indices)
        dsys.to("deepmd/npy", npy_temp_dir, set_size=settings["batch_size"])
        LOGGER.info("    Finished %s", base_name)
        return npy_temp_dir
    except Exception as e:
        LOGGER.error("Error processing %s: %s", outcar_file, e)
        return None

def process_all_outcar_files():
    """Process all OUTCAR files and gather temporary directories."""
    outcar_files = sorted(glob.glob(INPUT_PATTERN), key=lambda x: int(x.split('_')[-1]))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not outcar_files:
        LOGGER.warning("No files matching pattern '%s' found.", INPUT_PATTERN)
        return []

    LOGGER.info("Found %d OUTCAR files to process.", len(outcar_files))

    temp_dirs = []
    worker_count = MAX_WORKERS or (os.cpu_count() or 1)
    LOGGER.info("Using %d worker(s) for processing.", worker_count)

    settings = {
        "output_dir": OUTPUT_DIR,
        "skip_interval": SKIP_INTERVAL,
        "batch_size": dp_batch_size,
        "steps_to_remove": STEPS_TO_REMOVE,
    }

    global WORKER_SETTINGS
    WORKER_SETTINGS = settings

    with multiprocessing.Pool(worker_count, initializer=_worker_init, initargs=(settings,)) as pool:
        for i, result in enumerate(pool.imap(process_single_outcar, outcar_files), 1):
            file_name = os.path.basename(outcar_files[i-1])
            if result:
                temp_dirs.append(result)
                LOGGER.info("Processed %d/%d: %s", i, len(outcar_files), file_name)
            else:
                LOGGER.warning("Skipped %d/%d: %s", i, len(outcar_files), file_name)

    return temp_dirs

def flatten_and_merge(temp_dirs):
    """Flatten all set.XXX directories into sequentially named set.000, set.001, ..."""
    os.makedirs(FINAL_DIR, exist_ok=True)

    if not temp_dirs:
        LOGGER.warning("No temporary directories to merge.")
        return

    LOGGER.info("Merging %d temporary directories into '%s'.", len(temp_dirs), FINAL_DIR)

    set_counter = 0  # Start numbering from set.000

    for t_index, temp_dir in enumerate(temp_dirs, 1):
        LOGGER.info("  Directory %d/%d: %s", t_index, len(temp_dirs), temp_dir)
        # Collect only subdirectories that match 'set.*'
        set_dirs = sorted(glob.glob(os.path.join(temp_dir, "set.*")))
        for src_dir in set_dirs:
            dest_dir = os.path.join(FINAL_DIR, f"set.{set_counter:03d}")
            os.makedirs(dest_dir, exist_ok=True)
            for item in os.listdir(src_dir):
                src_item = os.path.join(src_dir, item)
                dest_item = os.path.join(dest_dir, item)
                shutil.move(src_item, dest_item)
            set_counter += 1

    LOGGER.info("Merged %d sets into '%s'.", set_counter, FINAL_DIR)

    # Copy type.raw and type_map.raw (assuming same for all)
    for filename in ["type.raw", "type_map.raw"]:
        src_file = os.path.join(temp_dirs[0], filename)
        if os.path.exists(src_file):
            shutil.copy2(src_file, FINAL_DIR)
            LOGGER.info("Copied %s into '%s'.", filename, FINAL_DIR)

def visualize_split(train_indices, val_indices, total_frames, output_file=None):
    """
    Visualize the train/validation split along the trajectory in two ways:
    1) Scatter with random vertical jitter (top).
    2) Histogram distribution of frame indices (bottom).

    A text box displays the number of frames in each split and the total.
    """

    train_count = len(train_indices)
    val_count = len(val_indices)

    # Create figure with 2 subplots sharing the x-axis
    fig, (ax_scatter, ax_hist) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # -----------------------------
    # 1) SCATTER WITH VERTICAL JITTER
    # -----------------------------
    # Random vertical offsets so points don't stack
    y_train = 0.1 + 0.05 * np.random.randn(train_count)
    y_val   = -0.1 + 0.05 * np.random.randn(val_count)

    ax_scatter.scatter(
        train_indices, y_train,
        color="blue", alpha=0.6, s=10, label="Train"
    )
    ax_scatter.scatter(
        val_indices, y_val,
        color="red", alpha=0.6, s=10, label="Val"
    )

    # Adjust the y-axis to keep points visible
    ax_scatter.set_ylim([-0.3, 0.3])
    ax_scatter.set_yticks([-0.1, 0.1])
    ax_scatter.set_yticklabels(["Val", "Train"])
    ax_scatter.set_ylabel("Random offset")
    ax_scatter.set_title("Train (blue) vs. Val (red) — Scatter w/ Jitter")
    ax_scatter.legend(loc="upper right")

    # Add text box with train/val counts
    textstr = (
        f"Train frames: {train_count}\n"
        f"Val frames:   {val_count}\n"
        f"Total frames: {train_count + val_count}"
    )
    ax_scatter.text(
        0.02, 0.98, textstr,
        transform=ax_scatter.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
    )

    # -----------------------------
    # 2) HISTOGRAM OF FRAME INDICES
    # -----------------------------
    # You can adjust the number of bins as you like:
    bins = max(10, total_frames // 50)

    ax_hist.hist(
        train_indices, bins=bins,
        alpha=0.5, color="blue", label="Train"
    )
    ax_hist.hist(
        val_indices, bins=bins,
        alpha=0.5, color="red", label="Val"
    )
    ax_hist.set_xlabel("Frame Index (consecutive)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Distribution of Frames by Index")
    ax_hist.legend(loc="upper right")

    # Final layout, save figure
    fig.tight_layout()
    if output_file is None:
        output_file = VISUALIZATION_FILE

    plt.savefig(output_file)
    plt.close()

    LOGGER.info("Visualization saved as %s", output_file)


def split_data():
    """Randomly split the merged deepmd data into training and validation sets,
    and visualize the splitting relative to the consecutive trajectory.
    """
    # Load the merged data from deepmd/npy format
    LOGGER.info("Loading merged data from '%s'...", FINAL_DIR)
    data = dpdata.LabeledSystem(FINAL_DIR, fmt="deepmd/npy")
    total_frames = len(data)
    LOGGER.info("Loaded %d frames. Splitting with train ratio %.2f.", total_frames, TRAIN_RATIO)
    
    # Create a random permutation of frame indices
    indices = np.random.permutation(total_frames)
    n_train = int(total_frames * TRAIN_RATIO)
    train_indices = np.sort(indices[:n_train])
    val_indices = np.sort(indices[n_train:])
    
    # Visualize the split
    if ENABLE_VISUALIZATION:
        visualize_split(train_indices, val_indices, total_frames)
    
    # Create sub-systems for train and validation data
    data_train = data.sub_system(train_indices)
    data_val = data.sub_system(val_indices)
    
    train_dir = os.path.join(FINAL_DIR, "train")
    val_dir = os.path.join(FINAL_DIR, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    LOGGER.info("Saving split datasets...")
    data_train.to_deepmd_npy(train_dir, set_size=dp_batch_size)
    data_val.to_deepmd_npy(val_dir, set_size=dp_batch_size)
    
    # Copy type files to the train and val directories if they exist
    for filename in ["type.raw", "type_map.raw"]:
        src_file = os.path.join(FINAL_DIR, filename)
        if os.path.exists(src_file):
            shutil.copy2(src_file, train_dir)
            shutil.copy2(src_file, val_dir)
    
    # Clean up the original merged content (only keep train/val directories)
    for item in os.listdir(FINAL_DIR):
        if item not in ["train", "val"]:
            path = os.path.join(FINAL_DIR, item)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    
    LOGGER.info(
        "Split data into %d training frames and %d validation frames.",
        len(data_train),
        len(data_val),
    )


def parse_arguments():
    """Parse command line arguments for the processing pipeline."""
    parser = argparse.ArgumentParser(
        description="Process VASP OUTCAR files into DeepMD datasets with optional splitting."
    )
    parser.add_argument("--input-pattern", default=INPUT_PATTERN, help="Glob pattern for OUTCAR files.")
    parser.add_argument("--steps-to-remove", type=int, default=STEPS_TO_REMOVE, help="Number of final steps to remove from each OUTCAR file.")
    parser.add_argument("--skip-interval", type=int, default=SKIP_INTERVAL, help="Use every Nth frame when generating data.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Temporary directory for intermediate DeepMD data.")
    parser.add_argument("--final-dir", default=FINAL_DIR, help="Directory that will hold the merged DeepMD data.")
    parser.add_argument("--log-file", default=LOG_FILE, help="Log file path.")
    parser.add_argument("--batch-size", type=int, default=dp_batch_size, help="Number of frames per DeepMD set.")
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO, help="Training split ratio (between 0 and 1).")
    parser.add_argument("--visualization-file", default=VISUALIZATION_FILE, help="Filename for the train/validation split visualization.")
    parser.add_argument("--no-visualization", action="store_true", help="Disable visualization generation.")
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum number of worker processes to spawn.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for deterministic splits.")
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep intermediate processed directories instead of deleting them.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output to warnings and errors.",
    )
    return parser.parse_args()


def apply_configuration(args):
    """Apply parsed arguments to global configuration variables."""
    global INPUT_PATTERN, STEPS_TO_REMOVE, SKIP_INTERVAL, OUTPUT_DIR, FINAL_DIR
    global LOG_FILE, dp_batch_size, TRAIN_RATIO, VISUALIZATION_FILE, MAX_WORKERS
    global ENABLE_VISUALIZATION

    INPUT_PATTERN = args.input_pattern
    STEPS_TO_REMOVE = args.steps_to_remove
    SKIP_INTERVAL = args.skip_interval
    OUTPUT_DIR = args.output_dir
    FINAL_DIR = args.final_dir
    LOG_FILE = args.log_file
    dp_batch_size = args.batch_size
    TRAIN_RATIO = args.train_ratio
    VISUALIZATION_FILE = args.visualization_file
    MAX_WORKERS = args.max_workers
    ENABLE_VISUALIZATION = not args.no_visualization

    if STEPS_TO_REMOVE < 0:
        raise ValueError("--steps-to-remove must be non-negative.")

    if SKIP_INTERVAL <= 0:
        raise ValueError("--skip-interval must be a positive integer.")

    if dp_batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")

    if not (0.0 < TRAIN_RATIO < 1.0):
        raise ValueError("--train-ratio must be between 0 and 1 (exclusive).")

    if MAX_WORKERS is not None and MAX_WORKERS <= 0:
        raise ValueError("--max-workers must be a positive integer when provided.")

    return args


def cleanup_intermediate(temp_dirs, keep_intermediate=False):
    """Remove intermediate directories unless retention was requested."""
    if keep_intermediate:
        LOGGER.info("Retaining intermediate directories.")
        return

    for directory in temp_dirs:
        if os.path.isdir(directory):
            shutil.rmtree(directory, ignore_errors=True)
            LOGGER.info("Removed intermediate directory: %s", directory)

    temp_root = os.path.abspath(OUTPUT_DIR)
    final_root = os.path.abspath(FINAL_DIR)

    if os.path.isdir(temp_root) and temp_root != final_root:
        shutil.rmtree(temp_root, ignore_errors=True)
        LOGGER.info("Removed temporary directory tree: %s", temp_root)

def main():
    args = parse_arguments()
    configure_logging(args.log_file, verbose=not args.quiet)

    try:
        apply_configuration(args)
    except ValueError as exc:
        LOGGER.error("Configuration error: %s", exc)
        sys.exit(2)

    if args.seed is not None:
        np.random.seed(args.seed)
        LOGGER.info("Random seed set to %d", args.seed)

    if args.keep_intermediate:
        LOGGER.warning("Intermediate directories will be retained as requested.")

    LOGGER.info("Big Chungus! === Step 1: Processing OUTCAR files ===")
    temp_dirs = process_all_outcar_files()

    if not temp_dirs:
        LOGGER.error("Big Chungus! No OUTCAR files were processed. Aborting.")
        sys.exit(1)

    LOGGER.info("Big Chungus! === Step 2: Merging into a single deepmd_data directory ===")
    flatten_and_merge(temp_dirs)

    LOGGER.info("Big Chungus! === Step 3: Splitting data into training and validation sets ===")
    split_data()

    cleanup_intermediate(temp_dirs, keep_intermediate=args.keep_intermediate)
    LOGGER.info("Big Chungus! Processing, merging, splitting, and visualization complete.")


if __name__ == '__main__':
    main()
