"""
3D CTA preprocessing and augmentation pipeline.

Deterministic preprocessing:
  - Standardize orientation + spacing
  - Image-based foreground/body crop (removes air/background)
  - Optional center crop / pad (approx heart ROI)
  - Configurable HU windowing + scaling
  - Optional edge-preserving denoise (TV-Chambolle)
  - Optional CLAHE (adaptive histogram equalization) slice-wise
  - Optional Frangi vesselness appended as an extra channel

Random augmentation (applied at runtime, not baked into cached outputs):
  - Geometric: flip, 90-degree rotation, affine, elastic
  - Intensity: contrast, shift, scale, histogram shift, Gaussian noise/blur
"""

from __future__ import annotations
import warnings

from typing import Sequence, Any
import numpy as np

from monai.config import KeysCollection
from monai.transforms.transform import MapTransform
from monai.transforms import (
    # Utility & Loading
    Compose,
    LoadImaged,
    EnsureTyped,
    EnsureChannelFirstd,

    # Spatial Transforms
    Orientationd,
    Spacingd,
    CropForegroundd,
    ResizeWithPadOrCropd,
    RandCropByPosNegLabeld,

    # Intensity Transforms
    ScaleIntensityRanged,
    NormalizeIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandHistogramShiftd,

    # Geometric Augmentations
    RandFlipd,
    RandRotate90d,
    RandAffined,
    Rand3DElasticd,
)

def _as_tuple3(x: Any, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if x is None:
        return default
    if isinstance(x, (list, tuple)) and len(x) == 3:
        return (int(x[0]), int(x[1]), int(x[2]))
    raise ValueError(f"Expected a 3-item list/tuple, got: {x}")


def _as_tuple2(x: Any, default: tuple[int, int]) -> tuple[int, int]:
    if x is None:
        return default
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return (int(x[0]), int(x[1]))
    raise ValueError(f"Expected a 2-item list/tuple, got: {x}")


class TVDenoised(MapTransform):
    """Edge-preserving 3D denoising using TV-Chambolle (anisotropic-diffusion-like).

    Works on NumPy arrays (MetaTensor is fine via np.asarray()).
    Input is expected to be float (after intensity scaling).

    This is a practical substitute for SimpleITK anisotropic diffusion when
    SimpleITK isn't available.
    """

    def __init__(
            self,
            keys: KeysCollection,
            weight: float = 0.06,
            max_num_iter: int = 20,
            eps: float = 2e-4,
    ) -> None:
        super().__init__(keys)
        self.weight = float(weight)
        self.max_num_iter = int(max_num_iter)
        self.eps = float(eps)
        try:
            from skimage.restoration import denoise_tv_chambolle  # noqa: F401
        except Exception as e:
            raise ImportError(
                "TVDenoised requires scikit-image. Install with: pip install scikit-image"
            ) from e

    def __call__(self, data: dict) -> dict:
        from skimage.restoration import denoise_tv_chambolle

        d = dict(data)
        for key in self.keys:
            img = d[key]
            arr = np.asarray(img).astype(np.float32)
            if arr.ndim != 4:
                raise ValueError(f"Expected image shape [C,H,W,D], got {arr.shape} for key={key}")
            out = np.empty_like(arr)
            for c in range(arr.shape[0]):
                out[c] = denoise_tv_chambolle(
                    arr[c],
                    weight=self.weight,
                    max_num_iter=self.max_num_iter,
                    eps=self.eps,
                    channel_axis=None,
                ).astype(np.float32)
            d[key] = out
        return d


class CLAHESlicewise3Dd(MapTransform):
    """Slice-wise CLAHE (adaptive histogram equalization).

    - Operates on [C,H,W,D] float volumes in [0,1].
    - Applies CLAHE on 2D slices along a chosen axis.

    Why slice-wise?
      Full 3D CLAHE is expensive and not widely available. Slice-wise CLAHE is a
      good compromise that often boosts small-vessel contrast.
    """

    def __init__(
            self,
            keys: KeysCollection,
            clip_limit: float = 0.01,
            kernel_size: int | None = None,
            axis: int = 3,
    ) -> None:
        super().__init__(keys)
        self.clip_limit = float(clip_limit)
        self.kernel_size = kernel_size
        self.axis = int(axis)
        try:
            from skimage import exposure  # noqa: F401
        except Exception as e:
            raise ImportError(
                "CLAHESlicewise3Dd requires scikit-image. Install with: pip install scikit-image"
            ) from e

    def __call__(self, data: dict) -> dict:
        from skimage import exposure

        d = dict(data)
        for key in self.keys:
            img = d[key]
            arr = np.asarray(img).astype(np.float32)
            if arr.ndim != 4:
                raise ValueError(f"Expected image shape [C,H,W,D], got {arr.shape} for key={key}")
            # clamp to [0,1]
            arr = np.clip(arr, 0.0, 1.0)
            out = np.empty_like(arr)

            # axis is in [0..3] for [C,H,W,D]; apply per-slice along spatial axis
            if self.axis not in (1, 2, 3):
                raise ValueError("axis must be 1(H), 2(W), or 3(D)")

            for c in range(arr.shape[0]):
                vol = arr[c]
                # Iterate slices
                if self.axis == 3:
                    for z in range(vol.shape[2]):
                        sl = vol[:, :, z]
                        out[c, :, :, z] = exposure.equalize_adapthist(
                            sl,
                            clip_limit=self.clip_limit,
                            kernel_size=self.kernel_size,
                        ).astype(np.float32)
                elif self.axis == 2:
                    for y in range(vol.shape[1]):
                        sl = vol[:, y, :]
                        out[c, :, y, :] = exposure.equalize_adapthist(
                            sl,
                            clip_limit=self.clip_limit,
                            kernel_size=self.kernel_size,
                        ).astype(np.float32)
                else:  # axis == 1
                    for x in range(vol.shape[0]):
                        sl = vol[x, :, :]
                        out[c, x, :, :] = exposure.equalize_adapthist(
                            sl,
                            clip_limit=self.clip_limit,
                            kernel_size=self.kernel_size,
                        ).astype(np.float32)

            d[key] = out
        return d


class FrangiVesselnessAsChanneld(MapTransform):
    """Compute Frangi vesselness on the (normalized) image and append as a channel.

    Output:
      image: [C+1, H, W, D]  (if keep_original=True)
         or [1, H, W, D]     (if keep_original=False)
    """

    def __init__(
            self,
            keys: KeysCollection,
            sigmas: Sequence[float] = (1.0, 2.0, 3.0, 4.0),
            alpha: float = 0.5,
            beta: float = 0.5,
            gamma: float = 15.0,
            black_ridges: bool = False,
            keep_original: bool = True,
    ) -> None:
        super().__init__(keys)
        self.sigmas = tuple(float(s) for s in sigmas)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.black_ridges = bool(black_ridges)
        self.keep_original = bool(keep_original)
        try:
            from skimage.filters import frangi  # noqa: F401
        except Exception as e:
            raise ImportError(
                "FrangiVesselnessAsChanneld requires scikit-image. Install with: pip install scikit-image"
            ) from e

    @staticmethod
    def _minmax01(x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        mn = float(x.min())
        mx = float(x.max())
        if mx <= mn + 1e-8:
            return np.zeros_like(x, dtype=np.float32)
        return (x - mn) / (mx - mn)

    def __call__(self, data: dict) -> dict:
        from skimage.filters import frangi

        d = dict(data)
        for key in self.keys:
            img = d[key]
            arr = np.asarray(img).astype(np.float32)
            if arr.ndim != 4:
                raise ValueError(f"Expected image shape [C,H,W,D], got {arr.shape} for key={key}")

            # For CTA, C is usually 1. If C>1, compute vesselness on channel 0.
            base = arr[0]
            base = np.clip(base, 0.0, 1.0)

            v = frangi(
                base,
                sigmas=self.sigmas,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                black_ridges=self.black_ridges,
            ).astype(np.float32)
            v = self._minmax01(v)
            v = v[None, ...]  # [1,H,W,D]

            if self.keep_original:
                out = np.concatenate([arr, v], axis=0)
            else:
                out = v
            d[key] = out
        return d


def get_train_transforms(
        patch_size: Sequence[int] = (96, 96, 96),
        num_samples: int = 4,
        preprocess_config: dict | None = None,
        augment_config: dict | None = None,
):
    """Training pipeline.

    `preprocess_config` and `augment_config` come from YAML under data.preprocess and data.augment.
    """
    preprocess_config = preprocess_config or {}
    augment_config = augment_config or {}

    keys = ["image", "label"]

    # -------------------------
    # Deterministic preprocessing
    # -------------------------
    pixdim = tuple(preprocess_config.get("pixdim", (1.0, 1.0, 1.0)))

    # HU windowing defaults tuned for contrast-enhanced CTA
    a_min = float(preprocess_config.get("window_a_min", -260.0))
    a_max = float(preprocess_config.get("window_a_max", 760.0))

    normalize_mode = str(preprocess_config.get("normalize_mode", "minmax")).lower()
    if normalize_mode not in ("minmax", "zscore", "minmax+zscore"):
        raise ValueError("normalize_mode must be one of: minmax, zscore, minmax+zscore")

    do_body_crop = bool(preprocess_config.get("body_crop", True))
    body_thresh = float(preprocess_config.get("body_threshold_hu", -500.0))
    body_margin = _as_tuple3(preprocess_config.get("body_margin", (8, 8, 8)), (8, 8, 8))

    do_center_crop = bool(preprocess_config.get("center_crop", False))
    center_crop_size = preprocess_config.get("center_crop_size", None)
    if do_center_crop and center_crop_size is None:
        # a conservative default (tune to your dataset)
        center_crop_size = (256, 256, 256)

    do_denoise = bool(preprocess_config.get("denoise_tv", True))
    denoise_weight = float(preprocess_config.get("denoise_weight", 0.06))
    denoise_iter = int(preprocess_config.get("denoise_n_iter", 20))

    do_clahe = bool(preprocess_config.get("clahe", False))
    clahe_clip = float(preprocess_config.get("clahe_clip_limit", 0.01))
    clahe_kernel = preprocess_config.get("clahe_kernel_size", None)  # int or None
    clahe_axis = int(preprocess_config.get("clahe_axis", 3))

    do_vesselness = bool(preprocess_config.get("vesselness", True))
    vesselness_as_channel = bool(preprocess_config.get("vesselness_as_channel", True))
    vesselness_keep_original = bool(preprocess_config.get("vesselness_keep_original", True))
    vesselness_sigmas = preprocess_config.get("vesselness_sigmas", (1, 2, 3, 4))
    vesselness_alpha = float(preprocess_config.get("vesselness_alpha", 0.5))
    vesselness_beta = float(preprocess_config.get("vesselness_beta", 0.5))
    vesselness_gamma = float(preprocess_config.get("vesselness_gamma", 15.0))

    deterministic = [
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=pixdim, mode=("bilinear", "nearest")),
    ]

    if do_body_crop:
        # Crop out air/background. This is image-only (no label leakage).
        deterministic.append(
            CropForegroundd(
                keys=keys,
                source_key="image",
                select_fn=lambda x: x > body_thresh,
                margin=body_margin,
            )
        )

    if do_center_crop:
        deterministic.append(
            ResizeWithPadOrCropd(keys=keys, spatial_size=_as_tuple3(center_crop_size, (256, 256, 256)))
        )

    # HU windowing to [0,1]
    deterministic.append(
        ScaleIntensityRanged(
            keys=["image"],
            a_min=a_min,
            a_max=a_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )
    )

    # Optional extra normalization
    if normalize_mode in ("zscore", "minmax+zscore"):
        deterministic.append(
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True)
        )

    if do_denoise:
        deterministic.append(
            TVDenoised(keys=["image"], weight=denoise_weight, max_num_iter=denoise_iter)
        )

    if do_clahe:
        deterministic.append(
            CLAHESlicewise3Dd(keys=["image"], clip_limit=clahe_clip, kernel_size=clahe_kernel, axis=clahe_axis)
        )

    if do_vesselness and vesselness_as_channel:
        deterministic.append(
            FrangiVesselnessAsChanneld(
                keys=["image"],
                sigmas=vesselness_sigmas,
                alpha=vesselness_alpha,
                beta=vesselness_beta,
                gamma=vesselness_gamma,
                black_ridges=False,
                keep_original=vesselness_keep_original,
            )
        )
    elif do_vesselness and not vesselness_as_channel:
        warnings.warn(
            "vesselness=True but vesselness_as_channel=False. No-op in this pipeline.",
            RuntimeWarning,
        )

    # -------------------------
    # Random augmentation (applied at runtime, not baked into cached output)
    # -------------------------
    aug = []

    # Existing geometric aug
    aug.extend(
        [
            RandFlipd(keys=keys, prob=float(augment_config.get("flip_prob", 0.5)), spatial_axis=0),
            RandFlipd(keys=keys, prob=float(augment_config.get("flip_prob", 0.5)), spatial_axis=1),
            RandFlipd(keys=keys, prob=float(augment_config.get("flip_prob", 0.5)), spatial_axis=2),
            RandRotate90d(keys=keys, prob=float(augment_config.get("rot90_prob", 0.5)), max_k=3, spatial_axes=(0, 1)),
        ]
    )

    # Stronger spatial aug (safe defaults)
    if bool(augment_config.get("rand_affine", True)):
        if RandAffined is None:
            warnings.warn("RandAffined is not available in your MONAI version; skipping affine augmentation.")
        else:
            aug.append(
                RandAffined(
                    keys=keys,
                    prob=float(augment_config.get("affine_prob", 0.3)),
                    rotate_range=tuple(augment_config.get("rotate_range", (0.15, 0.15, 0.15))),  # radians
                    translate_range=tuple(augment_config.get("translate_range", (8, 8, 8))),
                    scale_range=tuple(augment_config.get("scale_range", (0.10, 0.10, 0.10))),
                    mode=("bilinear", "nearest"),
                    padding_mode="border",
                )
            )

    if bool(augment_config.get("rand_elastic", False)):
        if Rand3DElasticd is None:
            warnings.warn("Rand3DElasticd is not available in your MONAI version; skipping elastic augmentation.")
        else:
            aug.append(
                Rand3DElasticd(
                    keys=keys,
                    prob=float(augment_config.get("elastic_prob", 0.15)),
                    sigma_range=tuple(augment_config.get("elastic_sigma_range", (4, 6))),
                    magnitude_range=tuple(augment_config.get("elastic_magnitude_range", (50, 90))),
                    rotate_range=tuple(augment_config.get("elastic_rotate_range", (0.05, 0.05, 0.05))),
                    translate_range=tuple(augment_config.get("elastic_translate_range", (6, 6, 6))),
                    scale_range=tuple(augment_config.get("elastic_scale_range", (0.05, 0.05, 0.05))),
                    mode=("bilinear", "nearest"),
                    padding_mode="border",
                )
            )

    # Intensity aug (very useful for CTA domain shift)
    if bool(augment_config.get("intensity_aug", True)):
        if RandAdjustContrastd is not None:
            aug.append(RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.8, 1.2)))
        else:
            warnings.warn("RandAdjustContrastd is not available in your MONAI version; skipping contrast augmentation.")

        if RandShiftIntensityd is not None:
            aug.append(RandShiftIntensityd(keys=["image"], prob=0.2, offsets=0.10))
        else:
            warnings.warn("RandShiftIntensityd is not available in your MONAI version; skipping intensity shift.")

        if RandScaleIntensityd is not None:
            aug.append(RandScaleIntensityd(keys=["image"], prob=0.2, factors=0.10))
        else:
            warnings.warn("RandScaleIntensityd is not available in your MONAI version; skipping intensity scale.")

        if RandHistogramShiftd is not None:
            aug.append(
                RandHistogramShiftd(
                    keys=["image"],
                    prob=0.15,
                    num_control_points=augment_config.get("hist_num_ctrl_pts", 10),
                )
            )
        else:
            warnings.warn("RandHistogramShiftd is not available in your MONAI version; skipping histogram shift.")

        if RandGaussianNoised is not None:
            aug.append(RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=float(augment_config.get("noise_std", 0.01))))
        else:
            warnings.warn("RandGaussianNoised is not available in your MONAI version; skipping Gaussian noise.")

        if RandGaussianSmoothd is not None:
            aug.append(RandGaussianSmoothd(keys=["image"], prob=0.10, sigma_x=(0.25, 1.0), sigma_y=(0.25, 1.0), sigma_z=(0.25, 1.0)))
        else:
            warnings.warn("RandGaussianSmoothd is not available in your MONAI version; skipping Gaussian smoothing aug.")

    # -------------------------
    # Patch extraction (class-balanced)
    # -------------------------
    sampler = [
        RandCropByPosNegLabeld(
            keys=keys,
            label_key="label",
            spatial_size=patch_size,
            pos=int(augment_config.get("pos_ratio", 1)),
            neg=int(augment_config.get("neg_ratio", 1)),
            num_samples=num_samples,
            image_key="image",
        ),
        EnsureTyped(keys=keys, device=None, track_meta=True),
    ]

    return Compose(deterministic + aug + sampler)


def get_val_transforms(
        preprocess_config: dict | None = None,
):
    """Validation/Test pipeline (deterministic only)."""
    preprocess_config = preprocess_config or {}

    keys = ["image", "label"]

    pixdim = tuple(preprocess_config.get("pixdim", (1.0, 1.0, 1.0)))
    a_min = float(preprocess_config.get("window_a_min", -260.0))
    a_max = float(preprocess_config.get("window_a_max", 760.0))

    normalize_mode = str(preprocess_config.get("normalize_mode", "minmax")).lower()
    if normalize_mode not in ("minmax", "zscore", "minmax+zscore"):
        raise ValueError("normalize_mode must be one of: minmax, zscore, minmax+zscore")

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

    do_clahe = bool(preprocess_config.get("clahe", False))
    clahe_clip = float(preprocess_config.get("clahe_clip_limit", 0.01))
    clahe_kernel = preprocess_config.get("clahe_kernel_size", None)
    clahe_axis = int(preprocess_config.get("clahe_axis", 3))

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
        Spacingd(keys=keys, pixdim=pixdim, mode=("bilinear", "nearest")),
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

    if do_clahe:
        t.append(CLAHESlicewise3Dd(keys=["image"], clip_limit=clahe_clip, kernel_size=clahe_kernel, axis=clahe_axis))

    if do_vesselness and vesselness_as_channel:
        t.append(
            FrangiVesselnessAsChanneld(
                keys=["image"],
                sigmas=vesselness_sigmas,
                alpha=vesselness_alpha,
                beta=vesselness_beta,
                gamma=vesselness_gamma,
                black_ridges=False,
                keep_original=vesselness_keep_original,
            )
        )

    t.append(EnsureTyped(keys=keys, device=None, track_meta=True))
    return Compose(t)
