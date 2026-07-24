# Inference Guide

Run trained models on new data to generate segmentation masks.

## Quick Start

```bash
python scripts/inference.py \
    --results_dir results/your_experiment_name \
    --input_dir /path/to/images \
    --output_dir /path/to/output
```

## Input/Output

- **Input:** Directory containing `*.img.nii.gz` files (or `*.img.nii`)
- **Output:** `*.label.nii.gz` files with matching names

Example:
```
input_dir/
  ├── patient_001.img.nii.gz
  ├── patient_002.img.nii.gz
  └── ...

output_dir/
  ├── patient_001.label.nii.gz
  ├── patient_002.label.nii.gz
  └── ...
```

## Options

### Model Specification

| Option | Description |
|--------|-------------|
| `--results_dir` | Path to experiment results (recommended) |
| `--config` | Alternative: path to config YAML |
| `--ckpt` | Checkpoint path (required if using `--config`) |

Using `--results_dir` automatically finds `config_snapshot.yaml` and the best checkpoint (`best_model.pt` or `best_model.pth`).

### Inference Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `--sw_batch_size` | 4 | Patches per batch (reduce if out of memory) |
| `--overlap` | 0.5 | Sliding window overlap (higher = smoother but slower) |
| `--mode` | gaussian | Aggregation mode: `gaussian` or `constant` |
| `--no_amp` | false | Disable mixed precision inference |

### Postprocessing

| Option | Default | Description |
|--------|---------|-------------|
| `--threshold` | 0.5 | Probability threshold for binarization |
| `--min_size` | 0 | Remove components smaller than N voxels (0 = disabled) |
| `--fg_class` | 1 | Foreground class index |

## Examples

**Basic inference:**
```bash
python scripts/inference.py \
    --results_dir results/att_mamba2_unet_baseline_batch4_lr1e-4 \
    --input_dir ./data/test_images \
    --output_dir ./predictions
```

**With tuned postprocessing:**
```bash
python scripts/inference.py \
    --results_dir results/att_mamba2_unet_baseline_batch4_lr1e-4 \
    --input_dir ./data/test_images \
    --output_dir ./predictions \
    --threshold 0.2 \
    --min_size 1000
```

**Low memory mode:**
```bash
python scripts/inference.py \
    --results_dir results/att_mamba2_unet_baseline_batch4_lr1e-4 \
    --input_dir ./data/test_images \
    --output_dir ./predictions \
    --sw_batch_size 1 \
    --no_amp
```

**Using explicit config and checkpoint:**
```bash
python scripts/inference.py \
    --config configs/unet_baseline.yaml \
    --ckpt results/unet_baseline/best_model.pth \
    --input_dir ./data/test_images \
    --output_dir ./predictions
```

## Postprocessing Tips

If you ran `tune_postprocessing_and_eval.py`, check `results_dir/postproc_tuning/best_postproc.json` for optimal parameters:

```bash
cat results/your_experiment/postproc_tuning/best_postproc.json
```

Then use those values:
```bash
python scripts/inference.py \
    --results_dir results/your_experiment \
    --input_dir ./data/test \
    --output_dir ./predictions \
    --threshold 0.2 \
    --min_size 1000
```
