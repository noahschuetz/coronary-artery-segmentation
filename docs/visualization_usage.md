# 3D Visualization Usage Guide

This guide explains how to use the 3D visualization system for evaluating coronary artery segmentation models.

## Quick Start

### 1. Install Dependencies

First, ensure you have the visualization dependencies installed:

```bash
pip install pyvista>=0.43.0 plotly>=5.18.0 kaleido>=0.2.1
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

### 2. Run Evaluation

Basic usage:

```bash
python scripts/evaluate_3d.py --experiment unet_baseline --top_k 5
```

This will:
- Load the trained model from `results/unet_baseline/`
- Run inference on the validation set
- Compute per-sample metrics
- Select the top 5 best and worst performing samples
- Generate 3D visualizations for each selected sample
- Create an HTML report with all results

## Command-Line Options

### Required Arguments

- `--experiment EXPERIMENT`: Name of the experiment directory in `results/`

### Optional Arguments

- `--top_k K`: Number of best/worst samples to visualize (default: 5)
- `--data_dir PATH`: Override data directory from config
- `--output_dir PATH`: Override output directory (default: `results/{experiment}/visualizations`)
- `--device {cuda,cpu}`: Device to use for inference (default: cuda)
- `--max_samples N`: Cap the number of validation samples processed (default: all)

## Usage Examples

### Example 1: Basic Evaluation

Evaluate the U-Net baseline model:

```bash
python scripts/evaluate_3d.py --experiment unet_baseline --top_k 5
```

### Example 2: Best-Performing Model

Evaluate the Attention-Mamba2 model (best performer):

```bash
python scripts/evaluate_3d.py --experiment att_mamba2_unet_baseline_batch4_lr1e-4 --top_k 3
```

### Example 3: Custom Data Directory

Use a different validation dataset:

```bash
python scripts/evaluate_3d.py --experiment unet_baseline --top_k 5 --data_dir ./data/university
```

### Example 4: CPU-Only Evaluation

Run on CPU (useful for remote servers without GPU):

```bash
python scripts/evaluate_3d.py --experiment unet_baseline --device cpu
```

### Example 5: Custom Output Directory

Save visualizations to a custom location:

```bash
python scripts/evaluate_3d.py --experiment unet_baseline --output_dir ./my_visualizations
```

## Output Structure

After running the evaluation, you'll find the following structure in the output directory:

```
results/{experiment}/visualizations/
├── samples/                          # Individual sample visualizations
│   ├── best_001_{patient_id}/
│   │   ├── volume_render_axial.png
│   │   ├── volume_render_sagittal.png
│   │   ├── volume_render_coronal.png
│   │   ├── volume_render_3d.png
│   │   ├── error_overlay.png         # Red=FP, Blue=FN, Green=TP
│   │   ├── uncertainty_map.png       # Prediction confidence
│   │   ├── comparison_panel.png      # 5-panel comprehensive view
│   │   └── interactive.html          # Rotatable 3D viewer
│   ├── best_002_{patient_id}/
│   ├── ...
│   ├── worst_001_{patient_id}/
│   └── worst_002_{patient_id}/
├── predictions_nifti/                # NIfTI predictions for external viewers
│   ├── {patient_id}_pred.nii.gz
│   └── ...
├── summary_report.html               # ← Open this in your browser!
└── metrics_per_sample.csv            # Detailed metrics table
```

## Viewing Results

### HTML Report

Open the main report in your web browser:

```bash
# Linux/Mac
xdg-open results/{experiment}/visualizations/summary_report.html

# Windows
start results/{experiment}/visualizations/summary_report.html
```

The HTML report includes:
- Executive summary with aggregate statistics
- Metrics distribution across all samples
- Gallery of best/worst performing samples
- Embedded visualizations
- Links to interactive 3D viewers
- Sortable metrics table

### Interactive 3D Viewers

Each sample has an `interactive.html` file that allows you to:
- Rotate and zoom the 3D visualization
- Toggle ground truth and prediction visibility
- View from different angles

### External Viewers

The NIfTI predictions can be loaded into professional medical imaging software:

- **ITK-SNAP**: Load CT image and overlay prediction
- **3D Slicer**: Advanced visualization and analysis
- **MITK Workbench**: Multi-modal visualization

## Visualization Types

### 1. Volume Renderings

Multi-view 3D renderings showing CT with segmentation overlays:
- **Axial view**: Top-down view
- **Sagittal view**: Side view
- **Coronal view**: Front view
- **3D view**: Oblique perspective

Colors:
- **Green**: Ground truth segmentation
- **Red**: Model prediction

### 2. Error Overlay

Color-coded error visualization:
- **Green**: True positives (correct predictions)
- **Red**: False positives (over-segmentation)
- **Blue**: False negatives (missed vessels)

### 3. Uncertainty Map

Heatmap showing prediction confidence:
- **Blue/Cool colors**: High confidence
- **Red/Hot colors**: Low confidence (uncertain regions)

Uses Shannon entropy of softmax probabilities.

### 4. Comparison Panel

5-panel comprehensive view showing:
1. CT volume alone
2. Ground truth overlay
3. Prediction overlay
4. Error overlay
5. Uncertainty map

All views use Maximum Intensity Projection (MIP) for clear visualization.

## Metrics Explained

### Volumetric Metrics

- **Dice**: Overlap between prediction and ground truth (0-1, higher is better)
- **IoU**: Intersection over Union (0-1, higher is better)
- **Precision**: Fraction of predicted vessel that is correct
- **Recall**: Fraction of true vessel that was predicted

### Topological Metrics

- **clDice**: Centerline Dice - measures vessel connectivity (0-1, higher is better)
- **Smooth clDice**: Distance-weighted centerline metric

### Distance Metrics

- **Hausdorff (95%)**: Maximum distance between predicted and true boundaries (mm, lower is better)

## Troubleshooting

### PyVista not available

If you see "PyVista not installed", the system will fall back to 2D slice visualization. To enable full 3D rendering:

```bash
pip install pyvista
```

### CUDA out of memory

If you encounter GPU memory errors, try:

1. Use CPU mode: `--device cpu`
2. Reduce the number of samples: `--top_k 2`

### No data files found

Ensure your data directory contains `.nii.gz` files in the expected format:
- `{patient_id}.img.nii.gz` (CT image)
- `{patient_id}.label.nii.gz` (segmentation label)

### Interactive HTML not working

Interactive 3D viewers require Plotly. Install with:

```bash
pip install plotly
```

## Performance Considerations

**Expected Runtime** (for 40 validation samples with top_k=5):
- Inference: 5-10 minutes (GPU)
- Metrics computation: 1-2 minutes
- Visualization generation: 5-10 minutes
- **Total**: ~15-20 minutes

**Tip**: Start with a small `--top_k` value (2-3) for quick testing.

## Advanced Usage

### Batch Processing

Evaluate multiple experiments:

```bash
for exp in unet_baseline unetr_baseline att_mamba2_unet_baseline_batch4_lr1e-4; do
    python scripts/evaluate_3d.py --experiment $exp --top_k 3
done
```

### Comparing Models

Generate visualizations for the same samples across different models by using consistent data directories and comparing the metrics CSV files.

## Next Steps

After generating visualizations:

1. **Review HTML report** to identify overall performance
2. **Analyze best performers** to understand what the model does well
3. **Study worst performers** to identify failure patterns
4. **Check error overlays** for systematic errors (e.g., missing branches)
5. **Examine uncertainty maps** to find regions where the model is uncertain
6. **Load NIfTI predictions** in ITK-SNAP for detailed analysis

