"""
Segmentation Evaluation Orchestrator

This module orchestrates the entire evaluation pipeline: loading models,
running inference, computing metrics, and generating visualizations.
"""

import os
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from monai.data import DataLoader

from src.models.model_factory import get_model
from src.data.dataset import _get_data_files
from src.data.transforms import get_val_transforms
from monai.data import Dataset

from .metrics_analyzer import compute_sample_metrics, create_metrics_table
from .volume_renderer import VolumeRenderer
from .error_visualization import compute_error_masks, render_error_overlay
from .uncertainty_maps import compute_entropy_map, visualize_uncertainty_mip
from .report_generator import generate_html_report


class SegmentationEvaluator:
    """
    Main evaluator for 3D segmentation visualization.

    Handles model loading, inference, metric computation, sample selection,
    and visualization generation.
    """

    def __init__(self, config: dict, model_path: str, data_dir: str, device: str = "cuda", max_samples: int = None):
        """
        Initialize evaluator.

        Args:
            config: Configuration dictionary (from YAML)
            model_path: Path to model checkpoint (.pth file)
            data_dir: Path to validation data directory
            device: Device to use ('cuda' or 'cpu')
            max_samples: Maximum number of samples to process (None = all)
        """
        self.config = config
        self.model_path = model_path
        self.data_dir = data_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_samples = max_samples

        # Extract config parameters
        self.model_config = config.get('model', {})
        self.data_config = config.get('data', {})
        self.patch_size = self.data_config.get('patch_size', [96, 96, 96])

        # Load model
        print(f"Loading model from {model_path}...")
        self.model = self._load_model()

        # Setup data loader
        print(f"Loading validation data from {data_dir}...")
        self.val_loader = self._setup_data_loader()

    def _load_model(self):
        """Load trained model from checkpoint."""
        # Instantiate model architecture
        model = get_model(self.model_config, self.data_config)

        # Load weights
        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(state_dict)

        # Move to device and set to eval mode
        model = model.to(self.device)
        model.eval()

        return model

    def _setup_data_loader(self):
        """Setup validation data loader."""
        # Get data files
        data_files = _get_data_files(self.data_dir)

        if not data_files:
            raise ValueError(f"No data files found in {self.data_dir}")

        # Limit number of samples if specified
        if self.max_samples is not None and self.max_samples < len(data_files):
            print(f"Limiting to first {self.max_samples} samples (out of {len(data_files)})")
            data_files = data_files[:self.max_samples]

        # Get validation transforms
        preprocess_config = self.data_config.get('preprocess', {})
        val_transforms = get_val_transforms(preprocess_config=preprocess_config)

        # Create dataset (no caching for evaluation)
        val_dataset = Dataset(data=data_files, transform=val_transforms)

        # Create data loader (batch size 1 for individual sample processing)
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,  # Single worker for sequential processing
        )

        print(f"Found {len(data_files)} validation samples.")

        return val_loader

    def run_inference(self):
        """
        Run sliding window inference on all validation samples.

        Returns:
            List of dicts, each containing:
                - patient_id: str
                - image: np.ndarray (H, W, D)
                - label: np.ndarray (H, W, D)
                - prediction: np.ndarray (H, W, D)
                - probabilities: np.ndarray (2, H, W, D)
                - spacing: tuple (sx, sy, sz)
                - affine: np.ndarray (4, 4)
        """
        results = []

        with torch.no_grad():
            for idx, val_data in enumerate(tqdm(self.val_loader, desc="Running inference")):
                # Move data to device
                inputs = val_data["image"].to(self.device)
                labels = val_data["label"].to(self.device)

                # Sliding window inference
                val_outputs = sliding_window_inference(
                    inputs=inputs,
                    roi_size=self.patch_size,
                    sw_batch_size=4,
                    predictor=self.model,
                    overlap=0.5,
                    mode="gaussian"
                )

                # Post-process
                probabilities = torch.softmax(val_outputs, dim=1)
                pred_classes = val_outputs.argmax(dim=1, keepdim=True)

                # Extract metadata
                try:
                    filename_or_obj = val_data["image_meta_dict"]["filename_or_obj"][0]
                    if isinstance(filename_or_obj, str):
                        patient_id = os.path.basename(filename_or_obj).split('.')[0]
                    else:
                        patient_id = str(filename_or_obj)
                except:
                    patient_id = f"sample_{len(results):03d}"

                # Extract spacing
                try:
                    spacing_tensor = val_data["image_meta_dict"]["pixdim"][0][1:4]
                    spacing = tuple(spacing_tensor.cpu().numpy())
                except:
                    spacing = (1.0, 1.0, 1.0)  # Default spacing

                # Extract affine matrix
                try:
                    affine = val_data["image_meta_dict"]["affine"][0].cpu().numpy()
                except:
                    affine = np.eye(4)

                # Store results
                results.append({
                    "index": idx,
                    "patient_id": patient_id,
                    "image": inputs[0, 0].cpu().numpy(),
                    "label": labels[0, 0].cpu().numpy(),
                    "prediction": pred_classes[0, 0].cpu().numpy(),
                    "probabilities": probabilities[0].cpu().numpy(),
                    "spacing": spacing,
                    "affine": affine,
                })

        return results

    def compute_per_sample_metrics(self, results: list):
        """
        Compute metrics for each sample.

        Args:
            results: List of result dicts from run_inference()

        Returns:
            pandas DataFrame with metrics, sorted by Dice score
        """
        print("Computing per-sample metrics...")
        metrics_list = []

        for result in tqdm(results, desc="Computing metrics"):
            metrics = compute_sample_metrics(
                pred=result["prediction"],
                label=result["label"],
                spacing=result["spacing"]
            )
            metrics["patient_id"] = result["patient_id"]
            metrics_list.append(metrics)

        df = create_metrics_table(metrics_list)

        return df

    def select_samples(self, metrics_df, results: list, top_k: int = 5):
        """
        Select best K and worst K samples for visualization.

        Args:
            metrics_df: DataFrame with metrics (sorted by Dice score descending)
            results: List of result dicts (same order as metrics_df)
            top_k: Number of best/worst samples to select

        Returns:
            List of tuples (category, rank, result_dict) where category is 'best' or 'worst'
        """
        print(f"Selecting top {top_k} best and worst samples...")

        # Use DataFrame indices to select samples
        # Best samples: first top_k rows (highest Dice scores)
        # Worst samples: last top_k rows (lowest Dice scores)
        best_indices = list(range(top_k))
        worst_indices = list(range(len(metrics_df) - top_k, len(metrics_df)))

        selected = []

        # Select best samples
        for i, idx in enumerate(best_indices):
            rank = i + 1
            result = results[idx]
            selected.append(("best", rank, result))

        # Select worst samples
        for i, idx in enumerate(worst_indices):
            rank = top_k - i  # Rank from worst (5) to less-worst (1)
            result = results[idx]
            selected.append(("worst", rank, result))

        return selected

    def generate_visualizations(self, selected_samples: list, output_dir: str):
        """
        Generate all visualizations for selected samples.

        Args:
            selected_samples: List of (category, rank, result_dict) tuples
            output_dir: Base output directory
        """
        print(f"Generating visualizations in {output_dir}...")

        # Create output directories
        samples_dir = os.path.join(output_dir, "samples")
        nifti_dir = os.path.join(output_dir, "predictions_nifti")
        os.makedirs(samples_dir, exist_ok=True)
        os.makedirs(nifti_dir, exist_ok=True)

        # Initialize renderer
        renderer = VolumeRenderer(theme='document', window_size=(1024, 768))

        for category, rank, result in tqdm(selected_samples, desc="Generating visualizations"):
            patient_id = result["patient_id"]
            sample_name = f"{category}_{rank:03d}_{patient_id}"
            sample_dir = os.path.join(samples_dir, sample_name)
            os.makedirs(sample_dir, exist_ok=True)

            # 1. Multi-view volume rendering
            print(f"  Rendering {sample_name}...")
            renderer.render_ct_with_overlay(
                ct_volume=result["image"],
                segmentation=result["label"],
                prediction=result["prediction"],
                spacing=result["spacing"],
                output_path=os.path.join(sample_dir, "volume_render.png"),
                view_angles=[
                    ('axial', 0, 90),
                    ('sagittal', 0, 0),
                    ('coronal', 90, 0),
                    ('3d', 45, 30),
                ]
            )

            # 2. Error overlay
            error_masks = compute_error_masks(result["prediction"], result["label"])
            render_error_overlay(
                ct_volume=result["image"],
                error_masks=error_masks,
                spacing=result["spacing"],
                output_path=os.path.join(sample_dir, "error_overlay.png"),
                use_mip=True
            )

            # 3. Uncertainty map
            entropy_map = compute_entropy_map(result["probabilities"])
            visualize_uncertainty_mip(
                ct_volume=result["image"],
                uncertainty_map=entropy_map,
                output_path=os.path.join(sample_dir, "uncertainty_map.png"),
                cmap='hot',
                alpha=0.7
            )

            # 4. Comparison panel
            error_rgb = self._create_error_rgb(error_masks)
            renderer.create_comparison_panel(
                ct_volume=result["image"],
                segmentation=result["label"],
                prediction=result["prediction"],
                uncertainty_map=entropy_map,
                error_rgb=error_rgb,
                output_path=os.path.join(sample_dir, "comparison_panel.png")
            )

            # 5. Interactive HTML
            renderer.create_interactive_html(
                ct_volume=result["image"],
                segmentation=result["label"],
                prediction=result["prediction"],
                spacing=result["spacing"],
                output_path=os.path.join(sample_dir, "interactive.html")
            )

            # 6. Save NIfTI prediction
            self._save_nifti(
                data=result["prediction"],
                affine=result["affine"],
                output_path=os.path.join(nifti_dir, f"{patient_id}_pred.nii.gz")
            )

    def _create_error_rgb(self, error_masks: dict) -> np.ndarray:
        """Create RGB visualization of errors."""
        from .error_visualization import create_error_rgb_map
        return create_error_rgb_map(error_masks)

    def _save_nifti(self, data: np.ndarray, affine: np.ndarray, output_path: str):
        """Save data as NIfTI file."""
        # Convert to uint8 for binary masks (compatible with medical imaging tools)
        data_uint8 = data.astype(np.uint8)
        nifti_img = nib.Nifti1Image(data_uint8, affine)
        nib.save(nifti_img, output_path)

    def run_full_evaluation(self, output_dir: str, top_k: int = 5):
        """
        Run complete evaluation pipeline.

        Args:
            output_dir: Output directory for all results
            top_k: Number of best/worst samples to visualize

        Returns:
            metrics_df: DataFrame with all sample metrics
        """
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Run inference
        print("\n[1/5] Running inference on validation set...")
        results = self.run_inference()

        # Step 2: Compute metrics
        print("\n[2/5] Computing per-sample metrics...")
        metrics_df = self.compute_per_sample_metrics(results)

        # Save metrics CSV
        metrics_csv_path = os.path.join(output_dir, "metrics_per_sample.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Metrics saved to {metrics_csv_path}")

        # Step 3: Select samples
        print(f"\n[3/5] Selecting top {top_k} best and worst samples...")
        selected = self.select_samples(metrics_df, results, top_k=top_k)

        # Step 4: Generate visualizations
        print(f"\n[4/5] Generating 3D visualizations...")
        self.generate_visualizations(selected, output_dir)

        # Step 5: Generate HTML report
        print("\n[5/5] Creating HTML report...")
        report_path = os.path.join(output_dir, "summary_report.html")
        generate_html_report(
            metrics_df=metrics_df,
            visualization_dir=output_dir,
            config=self.config,
            output_path=report_path,
            selected_samples=selected
        )

        print(f"\nEvaluation complete.")
        print(f"  Open {report_path} to view results.")

        return metrics_df
