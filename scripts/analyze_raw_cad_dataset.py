#!/usr/bin/env python3
"""
Raw CAD (coronary artery) dataset analysis (NO preprocessing).

Run:
  python scripts/analyze_raw_cad_dataset.py --config configs/att_mamba2_unet.yaml

Outputs:
  <results_dir>/raw_data_analysis/
    - per_case_stats.csv
    - summary.yaml
    - plots_*.png

Notes:
  - Uses same file discovery assumptions as src/data/dataset.py
  - Reads NIfTI via MONAI LoadImage (whatever reader MONAI has available in your env)
"""

import os
import csv
import math
import yaml
import argparse
import numpy as np

from monai.transforms import LoadImage

# reuse your dataset scanning logic (same assumptions about folder structure)
from src.data.dataset import _get_data_files  # noqa: E402

# optional CC stats
try:
    import scipy.ndimage as ndi
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# plotting (optional)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


def _to_np(x):
    """MONAI LoadImage may return numpy or torch; normalize to numpy array."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _get_affine(meta: dict):
    if not isinstance(meta, dict):
        return None
    aff = meta.get("affine", None)
    if aff is None:
        aff = meta.get("original_affine", None)
    if aff is None:
        return None
    return np.asarray(aff)


def _get_spacing(meta: dict):
    """
    Prefer meta['pixdim'] if present, else derive from affine column norms.
    Returns (sx, sy, sz) or (nan,nan,nan).
    """
    if isinstance(meta, dict):
        pixdim = meta.get("pixdim", None)
        if pixdim is not None and len(pixdim) >= 4:
            # MONAI often stores pixdim like (1, sx, sy, sz, ...)
            sx, sy, sz = float(pixdim[1]), float(pixdim[2]), float(pixdim[3])
            return sx, sy, sz

    aff = _get_affine(meta)
    if aff is None or aff.shape != (4, 4):
        return float("nan"), float("nan"), float("nan")
    R = aff[:3, :3]
    # spacing = norm of each column
    sx, sy, sz = np.sqrt((R * R).sum(axis=0)).tolist()
    return float(sx), float(sy), float(sz)


def _axis_permutation_and_flips(meta: dict):
    """
    Very lightweight orientation diagnostic:
      - estimates which voxel axis maps mostly to world x/y/z (permutation)
      - estimates sign flips (based on affine columns)
    """
    aff = _get_affine(meta)
    if aff is None or aff.shape != (4, 4):
        return None, None

    R = aff[:3, :3]
    col_norms = np.sqrt((R * R).sum(axis=0)) + 1e-8
    Rn = R / col_norms  # normalized columns

    # for each world axis (row), which voxel axis contributes most?
    # permutation of voxel axes to world axes
    perm = np.argmax(np.abs(Rn), axis=1).tolist()

    # sign of mapping for each world axis
    signs = []
    for world_axis in range(3):
        voxel_axis = perm[world_axis]
        signs.append(int(np.sign(Rn[world_axis, voxel_axis])))

    return perm, signs


def _percentiles(arr, ps):
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"p{int(p)}": float("nan") for p in ps}
    return {f"p{int(p)}": float(np.percentile(arr, p)) for p in ps}


def _connected_components(mask, connectivity=26):
    """
    mask: boolean or 0/1 ndarray (H,W,D)
    returns: (n_components, sizes_sorted_desc)
    """
    if not _HAS_SCIPY:
        return None, None

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
    else:
        st = np.ones((3, 3, 3), dtype=np.uint8)

    lab, n = ndi.label(mask.astype(bool), structure=st)
    if n == 0:
        return 0, []
    sizes = np.bincount(lab.ravel()).astype(np.int64)
    sizes[0] = 0
    sizes = sizes[sizes > 0]
    sizes_sorted = np.sort(sizes)[::-1].tolist()
    return int(n), sizes_sorted


def _bbox_from_mask(mask):
    """Returns bbox (minz,maxz,miny,maxy,minx,maxx) in index coords, or None if empty."""
    idx = np.argwhere(mask)
    if idx.size == 0:
        return None
    mins = idx.min(axis=0)
    maxs = idx.max(axis=0)
    # idx comes in (z,y,x) if mask is (Z,Y,X) OR (H,W,D) ??? so we keep generic:
    return mins.tolist(), maxs.tolist()


def _normalize_hu_window(img, a_min=-1024, a_max=400):
    """Matches your current ScaleIntensityRanged window (but done numerically here)."""
    x = np.clip(img, a_min, a_max)
    x = (x - a_min) / float(a_max - a_min)
    return x


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def plot_hist(values, out_path, title, xlabel, bins=50):
    if not _HAS_MPL:
        return
    v = np.asarray(values, dtype=np.float32)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    plt.figure()
    plt.hist(v, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser("Analyze raw CAD segmentation data (no preprocessing).")
    ap.add_argument("--config", required=True, help="Path to experiment YAML config.")
    ap.add_argument("--connectivity", type=int, default=26, choices=[6, 18, 26])
    ap.add_argument("--max_cases", type=int, default=0, help="0 = all, else limit for quick runs.")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    results_dir = config["results_dir"]
    out_dir = os.path.join(results_dir, "raw_data_analysis")
    os.makedirs(out_dir, exist_ok=True)

    # resolve data_dir relative to project root if needed
    data_dir = config["data"]["data_dir"]
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(ROOT_DIR, data_dir)

    patch_size = tuple(config["data"]["patch_size"])

    print(f"Data dir:     {data_dir}")
    print(f"Results dir:  {results_dir}")
    print(f"Patch size:   {patch_size}")
    print(f"Connectivity: {args.connectivity}")
    print(f"scipy CC:     {'yes' if _HAS_SCIPY else 'no (will skip CC stats)'}")
    print(f"matplotlib:   {'yes' if _HAS_MPL else 'no (will skip plots)'}")

    files = _get_data_files(data_dir=data_dir)
    if args.max_cases and args.max_cases > 0:
        files = files[: int(args.max_cases)]

    loader = LoadImage(image_only=False)

    per_case_rows = []
    spacing_list = []
    fg_frac_list = []
    fg_vox_list = []
    cc_count_list = []
    cc_largest_list = []
    label_hu_p50 = []
    label_hu_p95 = []
    label_hu_p99 = []

    # CSV header
    csv_path = os.path.join(out_dir, "per_case_stats.csv")
    header = [
        "case_idx",
        "image_path",
        "label_path",
        "shape_z",
        "shape_y",
        "shape_x",
        "sx",
        "sy",
        "sz",
        "perm_world_from_voxel",
        "signs_world",
        "label_unique_values",
        "fg_voxels",
        "fg_fraction",
        "bbox_mins",
        "bbox_maxs",
        "bbox_size_vox",
        "bbox_size_mm",
        "bbox_fits_patch",
        "img_min",
        "img_max",
        "img_mean",
        "img_std",
        "img_p1",
        "img_p50",
        "img_p99",
        "lbl_hu_mean",
        "lbl_hu_std",
        "lbl_hu_p1",
        "lbl_hu_p50",
        "lbl_hu_p95",
        "lbl_hu_p99",
        "lbl_norm_p50",  # after [-1024,400] -> [0,1]
        "cc_n",
        "cc_largest",
        "cc_p50",
        "cc_p95",
    ]

    print(f"\nAnalyzing {len(files)} cases...\n")

    for i, rec in enumerate(files):
        img_path = rec["image"]
        lbl_path = rec["label"]

        img, img_meta = loader(img_path)
        lbl, lbl_meta = loader(lbl_path)

        img_np = _to_np(img).astype(np.float32)
        lbl_np = _to_np(lbl)

        # Ensure we analyze single-channel volumes (common: (1,H,W,D) etc.)
        # We'll try to squeeze channel dims.
        img_np = np.squeeze(img_np)
        lbl_np = np.squeeze(lbl_np)

        # If user data is stored as (H,W,D), treat last dim as z-ish.
        # We'll standardize to (Z,Y,X) ordering for stats by moving axes if needed.
        # Heuristic: if 3D, accept as-is. We only use sizes and masks anyway.
        if img_np.ndim != 3 or lbl_np.ndim != 3:
            print(f"[WARN] Unexpected dims case {i}: img {img_np.shape}, lbl {lbl_np.shape}. Squeezing may have failed.")

        # Shape
        shape = img_np.shape
        # We'll label axes as (z,y,x) just for reporting
        shape_z, shape_y, shape_x = (shape[0], shape[1], shape[2]) if len(shape) == 3 else (math.nan, math.nan, math.nan)

        # Spacing / orientation diagnostics
        sx, sy, sz = _get_spacing(img_meta)
        perm, signs = _axis_permutation_and_flips(img_meta)

        # Label interpretation: treat any >0 as foreground, but also record unique values
        uniq = np.unique(lbl_np[~np.isnan(lbl_np)]).tolist()
        fg_mask = (lbl_np > 0)

        fg_vox = int(fg_mask.sum())
        total_vox = int(np.prod(lbl_np.shape)) if lbl_np.ndim == 3 else 0
        fg_frac = (fg_vox / total_vox) if total_vox > 0 else float("nan")

        # Bounding box
        bbox = _bbox_from_mask(fg_mask)
        if bbox is None:
            bbox_mins, bbox_maxs = None, None
            bbox_size_vox = None
            bbox_size_mm = None
            bbox_fits_patch = False
        else:
            bbox_mins, bbox_maxs = bbox
            # bbox sizes along each axis
            size_vox = [int(bbox_maxs[d] - bbox_mins[d] + 1) for d in range(3)]
            bbox_size_vox = size_vox
            # mm using spacing (best-effort)
            bbox_size_mm = [
                _safe_float(size_vox[2] * sx),  # x uses sx
                _safe_float(size_vox[1] * sy),  # y uses sy
                _safe_float(size_vox[0] * sz),  # z uses sz
            ]
            # Does bbox fit into patch (in voxels)? (heuristic)
            bbox_fits_patch = all(size_vox[d] <= patch_size[d] for d in range(3)) if len(patch_size) == 3 else False

        # Intensity stats (whole image)
        img_finite = img_np[np.isfinite(img_np)]
        img_min = float(np.min(img_finite)) if img_finite.size else float("nan")
        img_max = float(np.max(img_finite)) if img_finite.size else float("nan")
        img_mean = float(np.mean(img_finite)) if img_finite.size else float("nan")
        img_std = float(np.std(img_finite)) if img_finite.size else float("nan")
        img_ps = _percentiles(img_finite, [1, 50, 99])
        img_p1, img_p50, img_p99 = img_ps["p1"], img_ps["p50"], img_ps["p99"]

        # Intensity stats inside label region (HU stats for vessel voxels)
        if fg_vox > 0:
            lbl_vals = img_np[fg_mask]
            lbl_vals = lbl_vals[np.isfinite(lbl_vals)]
        else:
            lbl_vals = np.asarray([], dtype=np.float32)

        lbl_hu_mean = float(np.mean(lbl_vals)) if lbl_vals.size else float("nan")
        lbl_hu_std = float(np.std(lbl_vals)) if lbl_vals.size else float("nan")
        lbl_ps = _percentiles(lbl_vals, [1, 50, 95, 99])
        lbl_hu_p1 = lbl_ps["p1"]
        lbl_hu_p50_v = lbl_ps["p50"]
        lbl_hu_p95_v = lbl_ps["p95"]
        lbl_hu_p99_v = lbl_ps["p99"]

        # Apply your current HU window numerically and report median inside label
        img_norm = _normalize_hu_window(img_np, a_min=-1024, a_max=400)
        lbl_norm_p50 = float(np.median(img_norm[fg_mask])) if fg_vox > 0 else float("nan")

        # Connected components
        cc_n = None
        cc_largest = None
        cc_p50 = None
        cc_p95 = None
        if _HAS_SCIPY:
            cc_n, sizes_desc = _connected_components(fg_mask, connectivity=int(args.connectivity))
            if sizes_desc is not None and len(sizes_desc) > 0:
                cc_largest = int(sizes_desc[0])
                cc_p50 = float(np.percentile(np.asarray(sizes_desc), 50))
                cc_p95 = float(np.percentile(np.asarray(sizes_desc), 95))
            else:
                cc_largest = 0
                cc_p50 = 0.0
                cc_p95 = 0.0

        row = [
            i,
            img_path,
            lbl_path,
            shape_z, shape_y, shape_x,
            sx, sy, sz,
            str(perm) if perm is not None else "",
            str(signs) if signs is not None else "",
            str(uniq),
            fg_vox,
            fg_frac,
            str(bbox_mins) if bbox_mins is not None else "",
            str(bbox_maxs) if bbox_maxs is not None else "",
            str(bbox_size_vox) if bbox_size_vox is not None else "",
            str(bbox_size_mm) if bbox_size_mm is not None else "",
            int(bool(bbox_fits_patch)),
            img_min, img_max, img_mean, img_std, img_p1, img_p50, img_p99,
            lbl_hu_mean, lbl_hu_std, lbl_hu_p1, lbl_hu_p50_v, lbl_hu_p95_v, lbl_hu_p99_v,
            lbl_norm_p50,
            cc_n if cc_n is not None else "",
            cc_largest if cc_largest is not None else "",
            cc_p50 if cc_p50 is not None else "",
            cc_p95 if cc_p95 is not None else "",
        ]
        per_case_rows.append(row)

        # aggregate lists
        spacing_list.append([sx, sy, sz])
        fg_frac_list.append(fg_frac)
        fg_vox_list.append(fg_vox)
        if cc_n is not None:
            cc_count_list.append(cc_n)
            cc_largest_list.append(cc_largest)
        if not math.isnan(lbl_hu_p50_v):
            label_hu_p50.append(lbl_hu_p50_v)
        if not math.isnan(lbl_hu_p95_v):
            label_hu_p95.append(lbl_hu_p95_v)
        if not math.isnan(lbl_hu_p99_v):
            label_hu_p99.append(lbl_hu_p99_v)

        if (i + 1) % 25 == 0:
            print(f"  done {i+1}/{len(files)}")

    # write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(per_case_rows)

    # summarize
    spacing_arr = np.asarray(spacing_list, dtype=np.float32)
    sx_arr = spacing_arr[:, 0]
    sy_arr = spacing_arr[:, 1]
    sz_arr = spacing_arr[:, 2]

    def stats_1d(x):
        x = np.asarray(x, dtype=np.float32)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return {}
        return {
            "n": int(x.size),
            "min": float(np.min(x)),
            "p10": float(np.percentile(x, 10)),
            "p50": float(np.percentile(x, 50)),
            "p90": float(np.percentile(x, 90)),
            "max": float(np.max(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
        }

    summary = {
        "num_cases_analyzed": int(len(per_case_rows)),
        "patch_size_config": list(patch_size),
        "spacing_stats_mm": {
            "sx": stats_1d(sx_arr),
            "sy": stats_1d(sy_arr),
            "sz": stats_1d(sz_arr),
        },
        "foreground_voxels": stats_1d(fg_vox_list),
        "foreground_fraction": stats_1d(fg_frac_list),
        "label_hu_percentiles_across_cases": {
            "p50": stats_1d(label_hu_p50),
            "p95": stats_1d(label_hu_p95),
            "p99": stats_1d(label_hu_p99),
        },
        "connected_components": {
            "enabled": bool(_HAS_SCIPY),
            "connectivity": int(args.connectivity),
            "num_components": stats_1d(cc_count_list) if _HAS_SCIPY else {},
            "largest_component_size_vox": stats_1d(cc_largest_list) if _HAS_SCIPY else {},
        },
        "notes": [
            "Use per_case_stats.csv to spot outliers: very small vessels, extreme spacing, fragmented labels.",
            "If median spacing is already < 1mm, resampling to 1mm may erase thin vessels.",
            "If label HU percentiles sit mostly above 400, your current window [-1024,400] may clip vessel contrast.",
        ],
    }

    summary_path = os.path.join(out_dir, "summary.yaml")
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.dump(summary, f, sort_keys=False)

    # plots
    if _HAS_MPL:
        plot_hist(sx_arr, os.path.join(out_dir, "plots_spacing_sx.png"), "Spacing sx distribution", "mm")
        plot_hist(sy_arr, os.path.join(out_dir, "plots_spacing_sy.png"), "Spacing sy distribution", "mm")
        plot_hist(sz_arr, os.path.join(out_dir, "plots_spacing_sz.png"), "Spacing sz distribution", "mm")

        plot_hist(fg_frac_list, os.path.join(out_dir, "plots_fg_fraction.png"), "Foreground fraction distribution", "fg vox / total vox")

        if _HAS_SCIPY:
            plot_hist(cc_count_list, os.path.join(out_dir, "plots_cc_count.png"), "Connected components count distribution", "num CC")
            plot_hist(cc_largest_list, os.path.join(out_dir, "plots_cc_largest.png"), "Largest CC size distribution", "voxels")

        plot_hist(label_hu_p50, os.path.join(out_dir, "plots_label_hu_p50.png"), "Label HU p50 across cases", "HU")
        plot_hist(label_hu_p95, os.path.join(out_dir, "plots_label_hu_p95.png"), "Label HU p95 across cases", "HU")
        plot_hist(label_hu_p99, os.path.join(out_dir, "plots_label_hu_p99.png"), "Label HU p99 across cases", "HU")

    print("\nDone.")
    print(f"Saved per-case CSV: {csv_path}")
    print(f"Saved summary YAML: {summary_path}")
    print(f"Output folder: {out_dir}")


if __name__ == "__main__":
    main()
