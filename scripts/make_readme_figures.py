"""
Generate README figures for the portfolio project.

1. Grouped bar chart comparing the two segmentation models across metrics,
   read directly from each experiment's test_metrics.yaml.
2. (Optional) 3D surface render of a predicted coronary mask via marching cubes.

Usage:
    python scripts/make_readme_figures.py
"""

import os
import glob

import yaml
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASSETS_DIR = "docs/assets"
RESULTS = {
    "3D U-Net": "results/unet_baseline/test_metrics.yaml",
    "Att-Mamba2 U-Net": "results/att_mamba2_unet_baseline/test_metrics.yaml",
}

# dataviz categorical slots 1 & 2 (validated colorblind-safe pair)
COLORS = {"3D U-Net": "#2a78d6", "Att-Mamba2 U-Net": "#eb6834"}
METRICS = [
    ("dice", "Dice"),
    ("cldice", "clDice"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("iou", "IoU"),
]
INK = "#0b0b0b"
MUTED = "#52514e"


def load_metrics():
    out = {}
    for name, path in RESULTS.items():
        with open(path) as f:
            data = yaml.safe_load(f)
        out[name] = data["test_metrics"]
    return out


def metrics_chart(metrics):
    labels = [lbl for _, lbl in METRICS]
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, (model, vals) in enumerate(metrics.items()):
        offset = (i - 0.5) * width
        heights = [vals[key] for key, _ in METRICS]
        bars = ax.bar(
            x + offset,
            heights,
            width * 0.92,
            label=model,
            color=COLORS[model],
            zorder=3,
        )
        for b, h in zip(bars, heights):
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + 0.012,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=MUTED,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score (test split)", fontsize=10, color=MUTED)
    ax.set_title(
        "Coronary segmentation: CNN vs. Attention–Mamba2 hybrid",
        fontsize=12,
        color=INK,
        pad=12,
    )
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(axis="y", color="#e5e5e2", linewidth=0.8, zorder=0)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#cfcfca")
    ax.tick_params(length=0, colors=MUTED)

    fig.tight_layout()
    out = os.path.join(ASSETS_DIR, "metrics_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def surface_render():
    """Render a 3D surface of the largest predicted coronary mask (best effort)."""
    preds = sorted(glob.glob("data/predictions/*.label.nii.gz"))
    if not preds:
        print("no predictions found; skipping 3D render")
        return
    try:
        import nibabel as nib
        from skimage import measure
        import pyvista as pv

        pv.OFF_SCREEN = True
    except Exception as e:  # pragma: no cover
        print(f"3D render skipped (missing deps): {e}")
        return

    # pick the prediction with the most foreground voxels
    best, best_vox, best_affine = None, -1, None
    for p in preds[:40]:
        img = nib.load(p)
        arr = np.asarray(img.dataobj)
        v = int((arr > 0).sum())
        if v > best_vox:
            best, best_vox, best_affine = arr, v, img.affine
    if best is None or best_vox < 100:
        print("no suitable prediction volume; skipping 3D render")
        return

    verts, faces, _, _ = measure.marching_cubes(
        (best > 0).astype(np.float32), level=0.5
    )
    faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64)
    mesh = pv.PolyData(verts, faces_pv).smooth(n_iter=30)

    pl = pv.Plotter(off_screen=True, window_size=(1100, 900))
    pl.set_background("white")
    pl.add_mesh(mesh, color="#eb6834", specular=0.3, smooth_shading=True)
    pl.camera_position = "iso"
    pl.reset_camera()
    pl.camera.zoom(1.6)
    out = os.path.join(ASSETS_DIR, "predicted_tree_3d.png")
    try:
        pl.screenshot(out, transparent_background=False)
        print(f"wrote {out}")
    except Exception as e:  # pragma: no cover
        print(f"3D screenshot failed (likely no GL in this env): {e}")


def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    metrics = load_metrics()
    metrics_chart(metrics)
    surface_render()


if __name__ == "__main__":
    main()
