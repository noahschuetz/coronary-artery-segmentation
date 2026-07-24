#!/usr/bin/env python3
"""
Tune postprocessing on VAL and evaluate on TEST (best model checkpoint).

Key idea:
- Model is fixed (best checkpoint).
- VAL is used to tune postprocessing hyperparams.
- TEST is evaluated once with the tuned postprocessor.

It runs one inference per case, then re-uses probability maps for grid search (no re-inference).

Outputs (in results_dir/postproc_tuning/):
- best_postproc.json
- val_test_postproc_report.csv

Run:
  python scripts/tune_postprocessing_and_eval.py --config configs/att_mamba2_unet.yaml

Tips:
- If you already know the winner from analysis (thr=0.2, min_size=1000), you can restrict grid:
    --thresholds 0.2 --min_sizes 1000
"""

import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import yaml
import numpy as np
import torch
from torch.amp import autocast
from monai.inferers import sliding_window_inference

# Fix for PyTorch 2.6+ with MONAI cached data
from monai.data.meta_tensor import MetaTensor
from monai.utils.enums import MetaKeys, SpaceKeys, TraceKeys
torch.serialization.add_safe_globals([MetaTensor, MetaKeys, SpaceKeys, TraceKeys])

from src.data.dataset import get_data_loaders
from src.models.model_factory import get_model

# ---- scipy for CC + fill holes ----
try:
    import scipy.ndimage as ndi
except Exception as e:
    raise RuntimeError("This script requires scipy. Install: pip install scipy") from e


# -------------------------
# checkpoint helpers
# -------------------------
def get_best_ckpt_path(results_dir: str, preferred: str = None) -> str:
    if preferred:
        p = preferred
        if not os.path.isabs(p):
            p = os.path.join(results_dir, p)
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


def load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        model.load_state_dict(obj["model_state_dict"])
    else:
        model.load_state_dict(obj)


# -------------------------
# tensor -> numpy volume
# -------------------------
def to_vol_np(x: torch.Tensor) -> np.ndarray:
    """
    x shapes:
      (1, 1, H, W, D) or (1, H, W, D) or (H, W, D)
    -> (H, W, D)
    """
    x = x.detach().cpu()
    if x.ndim == 5:
        x = x[0, 0]
    elif x.ndim == 4:
        x = x[0]
    return x.numpy()


# -------------------------
# metrics
# -------------------------
def dice_from_masks(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum(dtype=np.int64)
    denom = pred.sum(dtype=np.int64) + gt.sum(dtype=np.int64)
    if denom == 0:
        return 1.0
    return float(2.0 * inter / (denom + 1e-8))


# -------------------------
# CC utilities
# -------------------------
def make_structure(connectivity: int) -> np.ndarray:
    """
    connectivity: 6, 18, 26
    """
    if connectivity == 6:
        st = np.zeros((3, 3, 3), dtype=np.uint8)
        st[1, 1, 1] = 1
        st[0, 1, 1] = st[2, 1, 1] = 1
        st[1, 0, 1] = st[1, 2, 1] = 1
        st[1, 1, 0] = st[1, 1, 2] = 1
        return st
    if connectivity == 18:
        st = np.ones((3, 3, 3), dtype=np.uint8)
        corners = [(0,0,0),(0,0,2),(0,2,0),(0,2,2),(2,0,0),(2,0,2),(2,2,0),(2,2,2)]
        for c in corners:
            st[c] = 0
        return st
    if connectivity == 26:
        return np.ones((3, 3, 3), dtype=np.uint8)
    raise ValueError("connectivity must be in {6,18,26}")


def label_and_sizes(mask: np.ndarray, connectivity: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Returns:
      lab: labeled volume
      sizes: bincount sizes for each label id (sizes[0]=background)
      n: number of components
    """
    st = make_structure(connectivity)
    lab, n = ndi.label(mask.astype(bool), structure=st)
    sizes = np.bincount(lab.ravel()).astype(np.int64)
    if sizes.size > 0:
        sizes[0] = 0
    return lab, sizes, int(n)


def mask_keep_lcc(lab: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    if sizes.size <= 1 or sizes.max() == 0:
        return np.zeros_like(lab, dtype=np.uint8)
    largest_id = int(np.argmax(sizes))
    return (lab == largest_id).astype(np.uint8)


def mask_keep_topk(lab: np.ndarray, sizes: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.zeros_like(lab, dtype=np.uint8)
    ids = np.where(sizes > 0)[0]
    if ids.size == 0:
        return np.zeros_like(lab, dtype=np.uint8)
    # sort label ids by size desc
    order = ids[np.argsort(sizes[ids])[::-1]]
    keep = order[:k]
    return np.isin(lab, keep).astype(np.uint8)


def mask_remove_small(lab: np.ndarray, sizes: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 0:
        return (lab > 0).astype(np.uint8)
    keep_ids = np.where(sizes >= int(min_size))[0]
    if keep_ids.size == 0:
        return np.zeros_like(lab, dtype=np.uint8)
    return np.isin(lab, keep_ids).astype(np.uint8)


def fill_holes_3d(mask: np.ndarray) -> np.ndarray:
    return ndi.binary_fill_holes(mask.astype(bool)).astype(np.uint8)


# -------------------------
# postproc config
# -------------------------
@dataclass(frozen=True)
class PostProcConfig:
    threshold: float
    method: str               # "none" | "lcc" | "topk" | "min_size"
    top_k: int = 0
    min_size: int = 0
    fill_holes: bool = False
    connectivity: int = 26

    def name(self) -> str:
        if self.method == "none":
            base = f"none_thr{self.threshold:.2f}"
        elif self.method == "lcc":
            base = f"lcc_thr{self.threshold:.2f}"
        elif self.method == "topk":
            base = f"topk{self.top_k}_thr{self.threshold:.2f}"
        elif self.method == "min_size":
            base = f"min{self.min_size}_thr{self.threshold:.2f}"
        else:
            base = f"{self.method}_thr{self.threshold:.2f}"
        if self.fill_holes:
            base += "_fill"
        base += f"_conn{self.connectivity}"
        return base


def apply_postproc(prob_fg: np.ndarray, cfg: PostProcConfig) -> np.ndarray:
    """
    prob_fg: (H,W,D) float
    returns mask uint8 (H,W,D)
    """
    bin_mask = (prob_fg >= float(cfg.threshold)).astype(np.uint8)

    if cfg.method == "none":
        out = bin_mask
    else:
        lab, sizes, n = label_and_sizes(bin_mask, cfg.connectivity)
        if cfg.method == "lcc":
            out = mask_keep_lcc(lab, sizes)
        elif cfg.method == "topk":
            out = mask_keep_topk(lab, sizes, cfg.top_k)
        elif cfg.method == "min_size":
            out = mask_remove_small(lab, sizes, cfg.min_size)
        else:
            raise ValueError(f"Unknown method: {cfg.method}")

    if cfg.fill_holes:
        out = fill_holes_3d(out)

    return out.astype(np.uint8)


# -------------------------
# inference (once per case)
# -------------------------
def infer_probs_for_loader(
        model: torch.nn.Module,
        loader,
        device: torch.device,
        patch_size: Tuple[int, int, int],
        fg_class: int,
        sw_batch_size: int,
        overlap: float,
        mode: str,
        use_amp: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Returns:
      probs_fg_list: list of (H,W,D) float32
      gt_mask_list:  list of (H,W,D) uint8
      filenames:     list of str
    """
    probs_list: List[np.ndarray] = []
    gts_list: List[np.ndarray] = []
    fns: List[str] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)  # (1,1,H,W,D)
            lbl = batch["label"].to(device)  # (1,1,H,W,D)
            meta = batch.get("image_meta_dict", None)
            fn = ""
            if isinstance(meta, dict):
                x = meta.get("filename_or_obj", None)
                if isinstance(x, (list, tuple)) and len(x) > 0:
                    fn = str(x[0])
                elif isinstance(x, str):
                    fn = x

            if (not use_amp) or device.type != "cuda":
                logits = sliding_window_inference(
                    inputs=img,
                    roi_size=patch_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                    mode=mode,
                )
            else:
                with autocast("cuda", dtype=torch.float16):
                    logits = sliding_window_inference(
                        inputs=img,
                        roi_size=patch_size,
                        sw_batch_size=sw_batch_size,
                        predictor=model,
                        overlap=overlap,
                        mode=mode,
                    )

            probs = torch.softmax(logits, dim=1)[:, fg_class:fg_class + 1]  # (1,1,H,W,D)
            prob_fg = to_vol_np(probs).astype(np.float32)                   # (H,W,D)
            gt = to_vol_np(lbl)                                             # (H,W,D)
            gt_mask = (gt == fg_class).astype(np.uint8)

            probs_list.append(prob_fg)
            gts_list.append(gt_mask)
            fns.append(fn)

    return probs_list, gts_list, fns


# -------------------------
# eval
# -------------------------
def mean_dice_over_dataset(
        probs_list: List[np.ndarray],
        gts_list: List[np.ndarray],
        cfg: PostProcConfig,
) -> float:
    dices = []
    for prob, gt in zip(probs_list, gts_list):
        pred = apply_postproc(prob.astype(np.float32), cfg)
        dices.append(dice_from_masks(pred, gt))
    return float(np.mean(dices)) if dices else float("nan")


def per_case_dice(
        probs_list: List[np.ndarray],
        gts_list: List[np.ndarray],
        cfg: PostProcConfig,
) -> List[float]:
    out = []
    for prob, gt in zip(probs_list, gts_list):
        pred = apply_postproc(prob.astype(np.float32), cfg)
        out.append(dice_from_masks(pred, gt))
    return out


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser("Tune postprocessing on VAL, evaluate on TEST")
    ap.add_argument("--config", required=True, help="Path to experiment YAML config")
    ap.add_argument("--ckpt", default=None, help="Checkpoint path override (optional)")

    ap.add_argument("--fg_class", type=int, default=1)
    ap.add_argument("--connectivity", type=int, default=26, choices=[6, 18, 26])

    ap.add_argument("--thresholds", type=float, nargs="*", default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7])
    ap.add_argument("--min_sizes", type=int, nargs="*", default=[200, 500, 800, 1000, 1500, 2000, 3000])
    ap.add_argument("--top_ks", type=int, nargs="*", default=[2, 3], help="Optional top-k CC keepers")
    ap.add_argument("--include_lcc", action="store_true", help="Also evaluate LCC (usually bad if GT has >1 CC)")
    ap.add_argument("--include_fill", action="store_true", help="Also evaluate fill-holes variants")

    ap.add_argument("--baseline_thr", type=float, default=0.5, help="Baseline threshold for reporting (no postproc)")
    ap.add_argument("--sw_batch_size", type=int, default=4)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--mode", type=str, default="gaussian", choices=["gaussian", "constant"])
    ap.add_argument("--no_amp", action="store_true")

    args = ap.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    results_dir = config["results_dir"]
    out_dir = os.path.join(results_dir, "postproc_tuning")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    patch_size = tuple(config["data"]["patch_size"])

    print(f"Device: {device}")
    print(f"Results dir: {results_dir}")
    print(f"Patch size: {patch_size}")

    # loaders
    _, val_loader, test_loader = get_data_loaders(config["data"], config["training"])
    if test_loader is None:
        raise RuntimeError("test_loader is None. Check test_split > 0 and data availability.")

    # model
    model = get_model(config["model"], config["data"]).to(device)
    ckpt = get_best_ckpt_path(results_dir, args.ckpt)
    print(f"Loading checkpoint: {ckpt}")
    load_checkpoint_into_model(model, ckpt, device)
    model.eval()

    use_amp = (not args.no_amp)

    print("\nInferring VAL probabilities...")
    val_probs, val_gts, val_files = infer_probs_for_loader(
        model=model,
        loader=val_loader,
        device=device,
        patch_size=patch_size,
        fg_class=int(args.fg_class),
        sw_batch_size=int(args.sw_batch_size),
        overlap=float(args.overlap),
        mode=str(args.mode),
        use_amp=use_amp,
    )

    print("Inferring TEST probabilities...")
    test_probs, test_gts, test_files = infer_probs_for_loader(
        model=model,
        loader=test_loader,
        device=device,
        patch_size=patch_size,
        fg_class=int(args.fg_class),
        sw_batch_size=int(args.sw_batch_size),
        overlap=float(args.overlap),
        mode=str(args.mode),
        use_amp=use_amp,
    )

    # baseline configs (reported)
    baseline_cfg = PostProcConfig(
        threshold=float(args.baseline_thr),
        method="none",
        fill_holes=False,
        connectivity=int(args.connectivity),
    )
    val_baseline = mean_dice_over_dataset(val_probs, val_gts, baseline_cfg)
    test_baseline = mean_dice_over_dataset(test_probs, test_gts, baseline_cfg)

    print("\n==================== BASELINE ====================")
    print(f"Baseline cfg: {baseline_cfg.name()}")
    print(f"VAL  mean Dice: {val_baseline:.4f}")
    print(f"TEST mean Dice: {test_baseline:.4f}")

    # build candidate configs for tuning on VAL
    candidates: List[PostProcConfig] = []
    conn = int(args.connectivity)

    for t in args.thresholds:
        t = float(t)
        # none
        candidates.append(PostProcConfig(threshold=t, method="none", connectivity=conn, fill_holes=False))
        # min_size
        for ms in args.min_sizes:
            candidates.append(PostProcConfig(threshold=t, method="min_size", min_size=int(ms), connectivity=conn, fill_holes=False))
        # top-k
        for k in args.top_ks:
            candidates.append(PostProcConfig(threshold=t, method="topk", top_k=int(k), connectivity=conn, fill_holes=False))
        # LCC optional
        if args.include_lcc:
            candidates.append(PostProcConfig(threshold=t, method="lcc", connectivity=conn, fill_holes=False))

    # optional fill-holes variants
    if args.include_fill:
        fill_cands = []
        for c in candidates:
            if c.method != "none":  # usually fill holes makes sense after CC filtering
                fill_cands.append(PostProcConfig(**{**asdict(c), "fill_holes": True}))
        candidates.extend(fill_cands)

    # Tune on VAL
    print("\n==================== TUNING ON VAL ====================")
    best_cfg = None
    best_val = -1.0

    # To keep it readable, we print whenever a new best is found
    for i, cfg in enumerate(candidates):
        m = mean_dice_over_dataset(val_probs, val_gts, cfg)
        if m > best_val:
            best_val = m
            best_cfg = cfg
            print(f"New best: VAL mean Dice={best_val:.4f}  cfg={best_cfg.name()}")

    if best_cfg is None:
        raise RuntimeError("No best config found (unexpected).")

    # Evaluate best on VAL + TEST
    val_best = mean_dice_over_dataset(val_probs, val_gts, best_cfg)
    test_best = mean_dice_over_dataset(test_probs, test_gts, best_cfg)

    print("\n==================== BEST POSTPROCESS ====================")
    print(f"Best cfg (tuned on VAL): {best_cfg.name()}")
    print(f"VAL  mean Dice: {val_best:.4f}  (delta vs baseline: {val_best - val_baseline:+.4f})")
    print(f"TEST mean Dice: {test_best:.4f}  (delta vs baseline: {test_best - test_baseline:+.4f})")

    # Per-case report CSV
    val_base_cases = per_case_dice(val_probs, val_gts, baseline_cfg)
    val_best_cases = per_case_dice(val_probs, val_gts, best_cfg)

    test_base_cases = per_case_dice(test_probs, test_gts, baseline_cfg)
    test_best_cases = per_case_dice(test_probs, test_gts, best_cfg)

    csv_path = os.path.join(out_dir, "val_test_postproc_report.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("split,case_idx,filename,dice_baseline,dice_postproc,delta\n")
        for i, (fn, b, p) in enumerate(zip(val_files, val_base_cases, val_best_cases)):
            fn_safe = (fn or "").replace(",", "_")
            f.write(f"val,{i},{fn_safe},{b:.6f},{p:.6f},{(p-b):+.6f}\n")
        for i, (fn, b, p) in enumerate(zip(test_files, test_base_cases, test_best_cases)):
            fn_safe = (fn or "").replace(",", "_")
            f.write(f"test,{i},{fn_safe},{b:.6f},{p:.6f},{(p-b):+.6f}\n")

    # Save chosen config + summary JSON
    out_json = {
        "baseline_cfg": asdict(baseline_cfg),
        "best_cfg": asdict(best_cfg),
        "val_mean_dice_baseline": val_baseline,
        "val_mean_dice_postproc": val_best,
        "test_mean_dice_baseline": test_baseline,
        "test_mean_dice_postproc": test_best,
        "val_delta": val_best - val_baseline,
        "test_delta": test_best - test_baseline,
    }
    json_path = os.path.join(out_dir, "best_postproc.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    print("\nSaved outputs:")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")



if __name__ == "__main__":
    main()
