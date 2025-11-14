# Playground-Codex

## CUTTail dpdata processing script

`CutTail_dpdata-process_v4.py` converts a collection of VASP `OUTCAR` files into
DeepMD-ready training and validation datasets. The script now provides:

- **Command-line configuration** for input patterns, batching, worker counts,
  logging, and visualization preferences.
- **Structured logging** to both the console and a log file so that
  multi-process runs remain easy to audit.
- **Optional cleanup** of intermediate artifacts generated while cleaning and
  batching the `OUTCAR` inputs, plus an opt-in JSON summary of each run.

### Quick start

```bash
python CutTail_dpdata-process_v4.py \
  --input-pattern "OUTCAR_???" \
  --steps-to-remove 10 \
  --skip-interval 2 \
  --output-dir processed_data \
  --final-dir deepmd_data \
  --batch-size 512 \
  --train-ratio 0.8
```

Additional helpful flags:

| Flag | Description |
| --- | --- |
| `--log-file` | Location of the detailed processing log. |
| `--max-workers` | Override the number of multiprocessing workers. |
| `--no-visualization` | Skip creation of the training/validation split plot. |
| `--visualization-file` | Override the filename used for the split visualization. |
| `--seed` | Make train/val splitting deterministic. |
| `--keep-intermediate` | Retain cleaned `OUTCAR` files and temporary DeepMD sets. |
| `--skip-split` | Finish after merging data without producing train/val sub-sets. |
| `--summary-file` | Path to a JSON file that captures run metadata and counts. |
| `--quiet` | Reduce console output to warnings and errors. |

Refer to `python CutTail_dpdata-process_v4.py --help` for the complete list of
options.
