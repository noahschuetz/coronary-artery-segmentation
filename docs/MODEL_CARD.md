---
license: mit
tags:
  - medical-imaging
  - segmentation
  - coronary-artery
  - 3d
  - ct
  - cta
  - unet
  - mamba
  - pytorch
  - monai
library_name: pytorch
pipeline_tag: image-segmentation
---

# Coronary Artery Segmentation (3D CTA) — U-Net vs. Attention–Mamba2

Two 3D segmentation models for **voxel-wise binary segmentation of the coronary artery tree** from cardiac CT angiography (CTA) volumes. Released as a portfolio study comparing a convolutional baseline against a state-space/attention hybrid encoder under an identical training and evaluation harness.

- **Repository:** https://github.com/noahschuetz/coronary-artery-segmentation
- **Task:** binary 3D semantic segmentation (foreground = coronary artery)
- **Framework:** PyTorch + MONAI
- **License:** MIT

## Checkpoints

| File | Model | Input channels | Params |
|---|---|---|---|
| `baseline_unet.pth` | 3D U-Net (MONAI) | 1 (CT) | ~31M |
| `att_mamba2_unet.pth` | Attention–Mamba2 U-Net | 2 (CT + Frangi vesselness) | ~31M |

`.pth` files are PyTorch state dicts. Instantiate the architecture from the repo, then load weights.

## Results

Identical training: AdamW (lr 1e-4, weight decay 1e-4, 5-epoch warmup), 100 epochs, 96³ patches, 4 patches/volume. Metrics on a held-out test split. clDice is topology-aware (penalizes centerline breaks).

| Model | Dice | clDice | Precision | Recall | IoU | Time/Epoch |
|---|---|---|---|---|---|---|
| 3D U-Net | 0.788 | 0.864 | 0.820 | 0.762 | 0.653 | ~1.7 min |
| Att-Mamba2 U-Net | 0.791 | 0.865 | 0.812 | 0.775 | 0.657 | ~1 min |

**Finding:** the two encoders are at parity on segmentation quality (~0.003 Dice apart). The state-space hybrid reaches that quality at roughly half the per-epoch training time. This is an efficiency/negative result, not a SOTA claim.

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from src.models.model_factory import get_model  # from the project repo

model_cfg = {"name": "att_mamba2_unet", "features": [32, 64, 128, 256], "in_channels": 2}
data_cfg = {"patch_size": [96, 96, 96]}
model = get_model(model_cfg, data_cfg)

ckpt = hf_hub_download("noahschuetz/coronary-segmentation", "att_mamba2_unet.pth")
state = torch.load(ckpt, map_location="cpu", weights_only=True)  # .pth is a plain state dict
model.load_state_dict(state)
model.eval()
```

Preprocessing must match training: reorient → resample → HU windowing → intensity normalization → (for the Att-Mamba2 model) append a Frangi vesselness channel. See `src/data/transforms.py`. Inference uses sliding-window patching over the full volume; see `scripts/inference.py`.

## Training data

Developed on the public **ImageCAS** coronary CTA dataset (cardiac CT angiography with expert coronary artery annotations). The dataset is not redistributed with these weights; obtain it from its original source.

## Intended use & limitations

- **Intended use:** research and educational demonstration of 3D vessel segmentation and architecture comparison.
- **Not for clinical use.** These models are not a medical device and have not been validated for diagnosis or treatment.
- **Domain shift:** trained on a single dataset's acquisition characteristics; performance will degrade on scanners, contrast protocols, or populations that differ from ImageCAS.
- **Thin-vessel failure modes:** distal/small branches and severe stenoses are the hardest cases; clDice is reported precisely because voxel Dice under-weights these topology errors.

## Citation

If you use these weights, please cite this repository and the ImageCAS dataset.
