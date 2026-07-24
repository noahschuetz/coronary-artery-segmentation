#!/usr/bin/env python3
"""
Standalone inference script for generating segmentation masks.

Usage:
  python scripts/inference.py \
      --results_dir results/att_mamba2_unet_baseline \
      --input_dir /path/to/images \
      --output_dir /path/to/output

  # Or with explicit config and checkpoint:
  python scripts/inference.py \
      --config configs/unet_baseline.yaml \
      --ckpt results/unet_baseline/best_model.pth \
      --input_dir /path/to/images \
      --output_dir /path/to/output

Input: Directory containing *.img.nii.gz files
Output: *.label.nii.gz files in the output directory
"""

import os
import glob
import argparse
from typing import Tuple, Optional

import yaml
import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm

# Fix for PyTorch 2.6+ with MONAI MetaTensor
from monai.data.meta_tensor import MetaTensor
from monai.utils.enums import MetaKeys, SpaceKeys, TraceKeys
torch.serialization.add_safe_globals([MetaTensor, MetaKeys, SpaceKeys, TraceKeys])

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    NormalizeIntensityd,
    CropForegroundd,
    ResizeWithPadOrCropd,
    EnsureTyped,
)
from monai.inferers import sliding_window_inference

from src.models.model_factory import get_model
from src.data.transforms import TVDenoised, FrangiVesselnessAsChanneld

# Optional scipy for postprocessing
try:
    import scipy.ndimage as ndi
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def get_checkpoint_path(results_dir: str, preferred: Optional[str] = None) -> str:
    """Find the best checkpoint in results directory."""
    if preferred:
        p = preferred if os.path.isabs(preferred) else os.path.join(results_dir, preferred)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    pt = os.path.join(results_dir, "best_model.pt")
    pth = os.path.join(results_dir, "best_model.pth")
    if os.path.exists(pt):
        return pt
    if os.path.exists(pth):
        return pth
    raise FileNotFoundError(f"No best_model.pt or best_model.pth found in: {results_dir}")


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    """Load checkpoint into model."""
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        model.load_state_dict(obj["model_state_dict"])
    else:
        model.load_state_dict(obj)


def _as_tuple3(x, default):
    if x is None:
        return default
    if isinstance(x, (list, tuple)) and len(x) == 3:
        return (int(x[0]), int(x[1]), int(x[2]))
    return default


def get_inference_transforms(preprocess_config: dict) -> Compose:
    """Build preprocessing transforms for inference (matches validation transforms)."""
    preprocess_config = preprocess_config or {}
    keys = ["image"]

    # Extract config with same defaults as get_val_transforms
    pixdim = tuple(preprocess_config.get("pixdim", (1.0, 1.0, 1.0)))
    a_min = float(preprocess_config.get("window_a_min", -260.0))
    a_max = float(preprocess_config.get("window_a_max", 760.0))

    normalize_mode = str(preprocess_config.get("normalize_mode", "minmax")).lower()

    do_body_crop = bool(preprocess_config.get("body_crop", True))
    body_thresh = float(preprocess_config.get("body_threshold_hu", -500.0))
    body_margin = _as_tuple3(preprocess_config.get("body_margin", (8, 8, 8)), (8, 8, 8))

    do_center_crop = bool(preprocess_config.get("center_crop", False))
    center_crop_size = preprocess_config.get("center_crop_size", None)
    if do_center_crop and center_crop_size is None:
        center_crop_size = (256, 256, 256)

    do_denoise = bool(preprocess_config.get("denoise_tv", True))
    denoise_weight = float(preprocess_config.get("denoise_weight", 0.06))
    denoise_iter = int(preprocess_config.get("denoise_n_iter", 20))

    do_vesselness = bool(preprocess_config.get("vesselness", True))
    vesselness_as_channel = bool(preprocess_config.get("vesselness_as_channel", True))
    vesselness_keep_original = bool(preprocess_config.get("vesselness_keep_original", True))
    vesselness_sigmas = preprocess_config.get("vesselness_sigmas", (1, 2, 3, 4))
    vesselness_alpha = float(preprocess_config.get("vesselness_alpha", 0.5))
    vesselness_beta = float(preprocess_config.get("vesselness_beta", 0.5))
    vesselness_gamma = float(preprocess_config.get("vesselness_gamma", 15.0))

    t = [
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=pixdim, mode="bilinear"),
    ]

    if do_body_crop:
        t.append(
            CropForegroundd(
                keys=keys,
                source_key="image",
                select_fn=lambda x: x > body_thresh,
                margin=body_margin,
            )
        )

    if do_center_crop:
        t.append(
            ResizeWithPadOrCropd(keys=keys, spatial_size=_as_tuple3(center_crop_size, (256, 256, 256)))
        )

    t.append(
        ScaleIntensityRanged(
            keys=["image"],
            a_min=a_min,
            a_max=a_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )
    )

    if normalize_mode in ("zscore", "minmax+zscore"):
        t.append(NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True))

    if do_denoise:
        t.append(TVDenoised(keys=["image"], weight=denoise_weight, max_num_iter=denoise_iter))

    if do_vesselness and vesselness_as_channel:
        t.append(
            FrangiVesselnessAsChanneld(
                keys=["image"],
                sigmas=vesselness_sigmas,
                alpha=vesselness_alpha,
                beta=vesselness_beta,
                gamma=vesselness_gamma,
                keep_original=vesselness_keep_original,
            )
        )

    t.append(EnsureTyped(keys=keys, data_type="tensor"))
    return Compose(t)


def apply_postprocessing(
    prob_map: np.ndarray,
    threshold: float = 0.5,
    min_size: int = 0,
    connectivity: int = 26,
) -> np.ndarray:
    """Apply threshold and optional connected component filtering."""
    mask = (prob_map >= threshold).astype(np.uint8)

    if min_size > 0 and HAS_SCIPY:
        # Build structure for connectivity
        if connectivity == 6:
            st = np.zeros((3, 3, 3), dtype=np.uint8)
            st[1, 1, 1] = 1
            st[0, 1, 1] = st[2, 1, 1] = 1
            st[1, 0, 1] = st[1, 2, 1] = 1
            st[1, 1, 0] = st[1, 1, 2] = 1
        elif connectivity == 18:
            st = np.ones((3, 3, 3), dtype=np.uint8)
            corners = [(0,0,0),(0,0,2),(0,2,0),(0,2,2),(2,0,0),(2,0,2),(2,2,0),(2,2,2)]
            for c in corners:
                st[c] = 0
        else:  # 26
            st = np.ones((3, 3, 3), dtype=np.uint8)

        labeled, num_features = ndi.label(mask, structure=st)
        if num_features > 0:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0  # Ignore background
            # Keep components larger than min_size
            keep_labels = np.where(sizes >= min_size)[0]
            mask = np.isin(labeled, keep_labels).astype(np.uint8)

    return mask


def run_inference(
    model: torch.nn.Module,
    image_path: str,
    transforms: Compose,
    device: torch.device,
    patch_size: Tuple[int, int, int],
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    mode: str = "gaussian",
    use_amp: bool = True,
    threshold: float = 0.5,
    min_size: int = 0,
    fg_class: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a single image.

    Returns:
        mask: Binary segmentation mask (H, W, D)
        affine: NIfTI affine matrix for saving
    """
    # Load and preprocess (dictionary-based transforms)
    data_dict = transforms({"image": image_path})
    image = data_dict["image"]

    # Get affine from metadata
    if hasattr(image, 'meta') and 'affine' in image.meta:
        affine = image.meta['affine']
    else:
        # Fallback: load original to get affine
        orig_nii = nib.load(image_path)
        affine = orig_nii.affine

    if isinstance(affine, torch.Tensor):
        affine = affine.numpy()

    # Add batch dimension: (C, H, W, D) -> (1, C, H, W, D)
    image = image.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = sliding_window_inference(
                    inputs=image,
                    roi_size=patch_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                    mode=mode,
                )
        else:
            logits = sliding_window_inference(
                inputs=image,
                roi_size=patch_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
                mode=mode,
            )

        # Get foreground probability
        probs = torch.softmax(logits, dim=1)
        prob_fg = probs[0, fg_class].cpu().numpy()  # (H, W, D)

    # Apply postprocessing
    mask = apply_postprocessing(prob_fg, threshold=threshold, min_size=min_size)

    return mask, affine


def save_nifti(mask: np.ndarray, affine: np.ndarray, output_path: str) -> None:
    """Save mask as NIfTI file."""
    nii = nib.Nifti1Image(mask.astype(np.uint8), affine)
    nib.save(nii, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a directory of NIfTI images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model/config specification (two ways)
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Path to results directory (contains config_snapshot.yaml and best_model.pth)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config YAML (alternative to --results_dir)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="Path to checkpoint (optional if using --results_dir)",
    )

    # Input/output
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing *.img.nii.gz files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save *.label.nii.gz files",
    )

    # Inference parameters
    parser.add_argument("--sw_batch_size", type=int, default=4, help="Sliding window batch size")
    parser.add_argument("--overlap", type=float, default=0.5, help="Sliding window overlap")
    parser.add_argument("--mode", type=str, default="gaussian", choices=["gaussian", "constant"])
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")

    # Postprocessing
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold")
    parser.add_argument("--min_size", type=int, default=0, help="Minimum connected component size (0=disabled)")
    parser.add_argument("--fg_class", type=int, default=1, help="Foreground class index")

    args = parser.parse_args()

    # Validate arguments
    if not args.results_dir and not args.config:
        parser.error("Must specify either --results_dir or --config")

    # Load config
    if args.results_dir:
        config_path = os.path.join(args.results_dir, "config_snapshot.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        ckpt_path = get_checkpoint_path(args.results_dir, args.ckpt)
    else:
        config_path = args.config
        if not args.ckpt:
            parser.error("Must specify --ckpt when using --config")
        ckpt_path = args.ckpt

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Setup
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    patch_size = tuple(config["data"]["patch_size"])
    preprocess_config = config["data"].get("preprocess", {})

    print(f"Device: {device}")
    print(f"Patch size: {patch_size}")
    print(f"Checkpoint: {ckpt_path}")

    # Build model
    model = get_model(config["model"], config["data"])
    load_checkpoint(model, ckpt_path, device)
    model = model.to(device)
    model.eval()
    print(f"Model loaded: {config['model']['name']}")

    # Build transforms
    transforms = get_inference_transforms(preprocess_config)

    # Find input files - support both direct files and directory structure
    image_entries = []  # List of (actual_path, patient_id)

    # 1. Check for *.img.nii.gz files (direct compressed files)
    for f in sorted(glob.glob(os.path.join(args.input_dir, "*.img.nii.gz"))):
        patient_id = os.path.basename(f).replace(".img.nii.gz", "")
        image_entries.append((f, patient_id))

    # 2. Check for *.img.nii (could be files OR directories)
    for p in sorted(glob.glob(os.path.join(args.input_dir, "*.img.nii"))):
        patient_id = os.path.basename(p).replace(".img.nii", "")
        # Skip if we already found a .gz version for this patient
        if any(pid == patient_id for _, pid in image_entries):
            continue

        if os.path.isdir(p):
            # Directory structure: find actual nii file inside
            for candidate in ["dia_0.nii", "diao_0.nii", "dia_0.nii.gz", "diao_0.nii.gz"]:
                candidate_path = os.path.join(p, candidate)
                if os.path.isfile(candidate_path):
                    image_entries.append((candidate_path, patient_id))
                    break
        else:
            # Direct uncompressed file
            image_entries.append((p, patient_id))

    if not image_entries:
        raise FileNotFoundError(f"No *.img.nii.gz or *.img.nii files/directories found in {args.input_dir}")

    print(f"Found {len(image_entries)} images to process")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each image
    for img_path, patient_id in tqdm(image_entries, desc="Running inference"):
        # Determine output filename
        out_name = f"{patient_id}.label.nii.gz"

        out_path = os.path.join(args.output_dir, out_name)

        # Run inference
        mask, affine = run_inference(
            model=model,
            image_path=img_path,
            transforms=transforms,
            device=device,
            patch_size=patch_size,
            sw_batch_size=args.sw_batch_size,
            overlap=args.overlap,
            mode=args.mode,
            use_amp=not args.no_amp,
            threshold=args.threshold,
            min_size=args.min_size,
            fg_class=args.fg_class,
        )

        # Save result
        save_nifti(mask, affine, out_path)

    print(f"\nDone! Saved {len(image_entries)} predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
