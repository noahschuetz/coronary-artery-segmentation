# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Medical imaging research project for **coronary artery segmentation from 3D CTA volumes**. It performs voxel-wise binary segmentation of the coronary artery tree and compares two architectures:

1. **3D U-Net** — MONAI-based convolutional baseline.
2. **Attention–Mamba2 U-Net** — hybrid encoder with multi-layer conv (MLC) stems, Mamba-2 state-space blocks, and windowed/global attention. A stride-2 stem reduces the volume 8× before attention stages. This is the current SOTA in this repo (best Dice, clDice, recall, IoU; ~1 min/epoch).

> Historical note: earlier commits also contained a graph/centerline-tracking pipeline (Trexplorer Super) and additional model variants (UNETR, Mamba-UNet, Mamba2-UNet). These were removed to focus the project on segmentation. Do not reference or reintroduce them.

## Environment & Installation

```bash
python -m venv mip_env && source mip_env/bin/activate
pip install -r requirements.txt
```

Key dependencies: PyTorch ≥2.0, MONAI, mamba-ssm, causal-conv1d, scikit-image, plotly/pyvista.

## Common Commands

### Training

```bash
# Att-Mamba2 U-Net (2 input channels: CT + Frangi vesselness)
python src/train.py --config configs/att_mamba2_unet.yaml

# 3D U-Net baseline (CT only)
python src/train.py --config configs/unet_baseline.yaml
```

### Inference & Evaluation

```bash
# Run inference on new volumes
python scripts/inference.py \
    --results_dir results/att_mamba2_unet_baseline \
    --input_dir ./data/test_images \
    --output_dir ./predictions

# Full 3D evaluation (Dice, HD95, clDice, Precision, Recall, F1, IoU)
python scripts/evaluate_3d.py \
    --predictions ./predictions \
    --ground_truth ./data/raw \
    --output_dir ./evaluation

# Sweep postprocessing hyperparameters (threshold, min component size, top-k)
python scripts/tune_postprocessing_and_eval.py \
    --config configs/att_mamba2_unet.yaml
```

There are no automated tests or a Makefile. Validation runs periodically during training, logged to `results/{experiment_name}/logs/` (TensorBoard).

## Architecture

### Data Format

Input CTA cases (default `data_dir` is `./data/imagecas`, configurable):
- `*.img.nii.gz` — CT volume (3D)
- `*.label.nii.gz` — Binary segmentation ground truth

`src/data/dataset.py` also supports a directory-per-case layout (`<id>.img.nii/` containing the volume, `<id>.label.nii/` containing the label). File discovery is in `_get_data_files`.

### Preprocessing

`src/data/transforms.py`: reorient → resample → HU windowing → intensity normalization → optional **Frangi vesselness** appended as a second channel (`FrangiVesselnessAsChanneld`, requires scikit-image). Enable/disable via `data.preprocess.vesselness` in the config. The Att-Mamba2 model expects 2 input channels (CT + vesselness); the baseline U-Net uses 1 (CT only).

### Segmentation Pipeline

```
CT Volume (96³ patch)
  → Segmentation Model → voxel probability map → threshold → connected components → binary mask
```

Models in `src/models/segmentation/`, selected via `model_factory.py`:
- `baseline_unet` — Standard MONAI 3D U-Net (`get_baseline_unet`).
- `att_mamba2_unet` — Hybrid Attention–Mamba2 U-Net (`get_att_mamba2_unet`), takes `patch_size` from the data config.

### Loss Functions

- `SoftCLDiceLoss` (`src/losses/soft_cldice_loss.py`) — topology-preserving; penalizes centerline/skeleton breaks in thin tubular structures.
- `DiceCECLDiceLoss` (`src/losses/combined_loss.py`) — combines MONAI `DiceCELoss` with `SoftCLDiceLoss` for both voxel accuracy and connectivity.

### Config-Driven Training

All parameters live in YAML configs (`configs/`):

```yaml
experiment_name: "att_mamba2_unet_baseline"
results_dir: "./results/att_mamba2_unet_baseline"
device: "cuda"
model:
  name: "att_mamba2_unet"     # maps to model_factory.py
  features: [32, 64, 128, 256]
  in_channels: 2              # CT + Frangi vesselness
data:
  data_dir: "./data/imagecas"
  val_split: 0.2
  patch_size: [96, 96, 96]
training:
  learning_rate: 1e-4
  weight_decay: 1e-4
  warmup_epochs: 5
  samples_per_volume: 4       # random patches per volume (DataLoader batch_size=1)
  num_epochs: 100
```

Results are saved to `results/{experiment_name}/` with the best checkpoint (`best_model.pt`/`.pth`), a config snapshot, TensorBoard logs, and `test_metrics.yaml`.

## Implementation Notes

- **PyTorch 2.6+ weights_only**: `src/train.py` sets `TORCH_FORCE_WEIGHTS_ONLY_LOAD=0` so MONAI's `PersistentDataset` cache (which pickles `MetaTensor` objects) loads correctly. Checkpoint loads pass `weights_only=False` explicitly.
- **clDice is slow** (CPU skeletonization): it is opt-in during training via `build_metrics_dict(include_cldice=...)` and intended mainly for final test evaluation.
- **Vesselness cache keying**: the `PersistentDataset` cache key accounts for `vesselness=True/False` so runs with different channel setups don't collide.
