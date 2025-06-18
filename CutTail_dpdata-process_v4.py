import dpdata
import glob
import os
import multiprocessing
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
# ========================

def clean_outcar(file_in, file_out, steps_to_remove=STEPS_TO_REMOVE):
    """Clean OUTCAR file by removing the last ``steps_to_remove`` steps.

    A ``steps_to_remove`` value of 0 keeps all steps intact. The function
    returns ``True`` if the output file was successfully written.
    """
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

def process_single_outcar(outcar_file, output_dir=OUTPUT_DIR):
    """Process a single OUTCAR file to DeepMD format."""
    base_name = os.path.basename(outcar_file)
    key = int(base_name.split('_')[-1])

    npy_temp_dir = os.path.join(output_dir, f"set_temp_{key:03d}")
    print(f"    Processing {base_name}...")
    outcar_clean = os.path.join(output_dir, f"{outcar_file}_clean")

    success = clean_outcar(outcar_file, outcar_clean)
    if not success:
        print(f"Failed to clean {base_name}. Skipping...")
        return None

    try:
        dsys = dpdata.LabeledSystem(outcar_clean, fmt="vasp/outcar")
        if SKIP_INTERVAL > 1:
            indices = range(0, dsys.get_nframes(), SKIP_INTERVAL)
            dsys = dsys.sub_system(indices)
        dsys.to("deepmd/npy", npy_temp_dir, set_size=dp_batch_size)
        print(f"    Finished {base_name}")
        return npy_temp_dir
    except Exception as e:
        print(f"Error processing {outcar_file}: {e}")
        return None

def process_all_outcar_files():
    """Process all OUTCAR files and gather temporary directories."""
    outcar_files = sorted(glob.glob(INPUT_PATTERN), key=lambda x: int(x.split('_')[-1]))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not outcar_files:
        print(f"No files matching pattern '{INPUT_PATTERN}' found.")
        return []

    print(f"Found {len(outcar_files)} OUTCAR files to process.")

    temp_dirs = []
    with multiprocessing.Pool(os.cpu_count() or 1) as pool:
        for i, result in enumerate(pool.imap(process_single_outcar, outcar_files), 1):
            file_name = os.path.basename(outcar_files[i-1])
            if result:
                temp_dirs.append(result)
                print(f"Processed {i}/{len(outcar_files)}: {file_name}")
            else:
                print(f"Skipped {i}/{len(outcar_files)}: {file_name}")

    return temp_dirs

def flatten_and_merge(temp_dirs, final_dir=FINAL_DIR):
    """Flatten all set.XXX directories into sequentially named set.000, set.001, ..."""
    os.makedirs(final_dir, exist_ok=True)

    if not temp_dirs:
        print("No temporary directories to merge.")
        return

    print(f"Merging {len(temp_dirs)} temporary directories into '{final_dir}'.")

    set_counter = 0  # Start numbering from set.000

    for t_index, temp_dir in enumerate(temp_dirs, 1):
        print(f"  Directory {t_index}/{len(temp_dirs)}: {temp_dir}")
        # Collect only subdirectories that match 'set.*'
        set_dirs = sorted(glob.glob(os.path.join(temp_dir, "set.*")))
        for src_dir in set_dirs:
            dest_dir = os.path.join(final_dir, f"set.{set_counter:03d}")
            os.makedirs(dest_dir, exist_ok=True)
            for item in os.listdir(src_dir):
                src_item = os.path.join(src_dir, item)
                dest_item = os.path.join(dest_dir, item)
                shutil.move(src_item, dest_item)
            set_counter += 1

    print(f"Merged {set_counter} sets into '{final_dir}'.")

    # Copy type.raw and type_map.raw (assuming same for all)
    for filename in ["type.raw", "type_map.raw"]:
        src_file = os.path.join(temp_dirs[0], filename)
        if os.path.exists(src_file):
            shutil.copy2(src_file, final_dir)

def visualize_split(train_indices, val_indices, total_frames, output_file=VISUALIZATION_FILE):
    """
    Visualize the train/validation split along the trajectory in two ways:
    1) Scatter with random vertical jitter (top).
    2) Histogram distribution of frame indices (bottom).

    A text box displays the number of frames in each split and the total.
    """
    import matplotlib.pyplot as plt

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
    plt.savefig(output_file)
    plt.close()

    print(f"Visualization saved as {output_file}")


def split_data(final_dir=FINAL_DIR, train_ratio=TRAIN_RATIO):
    """Randomly split the merged deepmd data into training and validation sets,
    and visualize the splitting relative to the consecutive trajectory.
    """
    # Load the merged data from deepmd/npy format
    print(f"Loading merged data from '{final_dir}'...")
    data = dpdata.LabeledSystem(final_dir, fmt="deepmd/npy")
    total_frames = len(data)
    print(f"Loaded {total_frames} frames. Splitting with train ratio {train_ratio}.")
    
    # Create a random permutation of frame indices
    indices = np.random.permutation(total_frames)
    n_train = int(total_frames * train_ratio)
    train_indices = np.sort(indices[:n_train])
    val_indices = np.sort(indices[n_train:])
    
    # Visualize the split
    visualize_split(train_indices, val_indices, total_frames)
    
    # Create sub-systems for train and validation data
    data_train = data.sub_system(train_indices)
    data_val = data.sub_system(val_indices)
    
    train_dir = os.path.join(final_dir, "train")
    val_dir = os.path.join(final_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    print("Saving split datasets...")
    data_train.to_deepmd_npy(train_dir, set_size=dp_batch_size)
    data_val.to_deepmd_npy(val_dir, set_size=dp_batch_size)
    
    # Copy type files to the train and val directories if they exist
    for filename in ["type.raw", "type_map.raw"]:
        src_file = os.path.join(final_dir, filename)
        if os.path.exists(src_file):
            shutil.copy2(src_file, train_dir)
            shutil.copy2(src_file, val_dir)
    
    # Clean up the original merged content (only keep train/val directories)
    for item in os.listdir(final_dir):
        if item not in ["train", "val"]:
            path = os.path.join(final_dir, item)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    
    print(f"Split data into {len(data_train)} training frames and {len(data_val)} validation frames.")

if __name__ == '__main__':
    print("Big Chungus! === Step 1: Processing OUTCAR files ===")
    temp_dirs = process_all_outcar_files()

    if not temp_dirs:
        print("Big Chungus! No OUTCAR files were processed. Aborting.")
        sys.exit(1)

    print("Big Chungus! === Step 2: Merging into a single deepmd_data directory ===")
    flatten_and_merge(temp_dirs)

    print("Big Chungus! === Step 3: Splitting data into training and validation sets ===")
    split_data()

    print("Big Chungus! Processing, merging, splitting, and visualization complete.")
