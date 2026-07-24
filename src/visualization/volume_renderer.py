"""
3D Volume Rendering using PyVista

This module provides high-quality 3D volume rendering capabilities for medical imaging,
including volume rendering, surface mesh extraction, and interactive HTML visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("Warning: PyVista not installed. 3D rendering will fall back to 2D slices.")

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not installed. Interactive HTML visualization unavailable.")


class VolumeRenderer:
    """
    3D volume renderer using PyVista for medical imaging visualization.
    """

    def __init__(self, theme: str = 'document', window_size: tuple = (1024, 768)):
        """
        Initialize volume renderer.

        Args:
            theme: PyVista theme ('document', 'dark', 'paraview')
            window_size: Rendering window size (width, height)
        """
        if PYVISTA_AVAILABLE:
            pv.set_plot_theme(theme)
        self.window_size = window_size

    def _normalize_ct_volume(self, ct_volume: np.ndarray,
                            window_min: float = -1024, window_max: float = 400) -> np.ndarray:
        """
        Apply CT windowing and normalize to [0, 1].

        Args:
            ct_volume: Raw CT data in Hounsfield Units
            window_min: Lower bound of CT window (default: -1024)
            window_max: Upper bound of CT window (default: 400)

        Returns:
            Normalized CT volume [0, 1]
        """
        # Clip to window
        ct_windowed = np.clip(ct_volume, window_min, window_max)

        # Normalize to [0, 1]
        ct_norm = (ct_windowed - window_min) / (window_max - window_min)

        return ct_norm.astype(np.float32)

    def _create_ct_opacity_map(self) -> str:
        """
        Create opacity transfer function for CT visualization.

        Returns:
            String opacity mapping compatible with PyVista
        """
        # Return linear opacity mapping as string
        # Format: "value:opacity value:opacity ..."
        # This creates a gradual opacity increase for higher density structures
        return "linear"

    def render_ct_with_overlay(self,
                               ct_volume: np.ndarray,
                               segmentation: np.ndarray,
                               prediction: np.ndarray,
                               spacing: tuple,
                               output_path: str,
                               view_angles: list = None):
        """
        Create multi-view rendering of CT with ground truth and prediction overlays using matplotlib.

        Args:
            ct_volume: 3D CT data in Hounsfield Units (H, W, D)
            segmentation: Ground truth binary mask (H, W, D)
            prediction: Prediction binary mask (H, W, D)
            spacing: Voxel spacing in mm (sx, sy, sz)
            output_path: Path to save PNG (should not include view suffix)
            view_angles: List of view names (not used, kept for API compatibility)
        """
        H, W, D = ct_volume.shape

        # Define views and their corresponding slices
        views = {
            'axial': ('Axial View (Z-axis)', ct_volume[:, :, D // 2],
                     segmentation[:, :, D // 2], prediction[:, :, D // 2]),
            'sagittal': ('Sagittal View (X-axis)', ct_volume[H // 2, :, :],
                        segmentation[H // 2, :, :], prediction[H // 2, :, :]),
            'coronal': ('Coronal View (Y-axis)', ct_volume[:, W // 2, :],
                       segmentation[:, W // 2, :], prediction[:, W // 2, :]),
        }

        # Render each orthogonal view
        for view_name, (title, ct_slice, gt_slice, pred_slice) in views.items():
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            # CT alone
            axes[0].imshow(ct_slice.T, cmap='gray', origin='lower')
            axes[0].set_title('CT Volume')
            axes[0].axis('off')

            # Ground truth overlay
            axes[1].imshow(ct_slice.T, cmap='gray', origin='lower')
            axes[1].imshow(gt_slice.T, cmap='Greens', alpha=0.5, origin='lower')
            axes[1].set_title('Ground Truth (Green)')
            axes[1].axis('off')

            # Prediction overlay
            axes[2].imshow(ct_slice.T, cmap='gray', origin='lower')
            axes[2].imshow(pred_slice.T, cmap='Reds', alpha=0.5, origin='lower')
            axes[2].set_title('Prediction (Red)')
            axes[2].axis('off')

            # Both overlays
            axes[3].imshow(ct_slice.T, cmap='gray', origin='lower')
            axes[3].imshow(gt_slice.T, cmap='Greens', alpha=0.4, origin='lower')
            axes[3].imshow(pred_slice.T, cmap='Reds', alpha=0.4, origin='lower')
            axes[3].set_title('Overlay (Green=GT, Red=Pred)')
            axes[3].axis('off')

            plt.suptitle(title, fontsize=14, fontweight='bold')
            plt.tight_layout()

            view_output_path = output_path.replace('.png', f'_{view_name}.png')
            plt.savefig(view_output_path, dpi=150, bbox_inches='tight')
            plt.close()

        # Create a "3D" view showing multiple slices at different depths
        num_slices = 3
        fig, axes = plt.subplots(num_slices, 4, figsize=(20, 5 * num_slices))

        for i in range(num_slices):
            slice_idx = int((i + 1) * D / (num_slices + 1))
            ct_slice = ct_volume[:, :, slice_idx]
            gt_slice = segmentation[:, :, slice_idx]
            pred_slice = prediction[:, :, slice_idx]

            # CT alone
            axes[i, 0].imshow(ct_slice.T, cmap='gray', origin='lower')
            axes[i, 0].set_title(f'CT Volume (Slice {slice_idx}/{D})')
            axes[i, 0].axis('off')

            # Ground truth overlay
            axes[i, 1].imshow(ct_slice.T, cmap='gray', origin='lower')
            axes[i, 1].imshow(gt_slice.T, cmap='Greens', alpha=0.5, origin='lower')
            axes[i, 1].set_title('Ground Truth (Green)')
            axes[i, 1].axis('off')

            # Prediction overlay
            axes[i, 2].imshow(ct_slice.T, cmap='gray', origin='lower')
            axes[i, 2].imshow(pred_slice.T, cmap='Reds', alpha=0.5, origin='lower')
            axes[i, 2].set_title('Prediction (Red)')
            axes[i, 2].axis('off')

            # Both overlays
            axes[i, 3].imshow(ct_slice.T, cmap='gray', origin='lower')
            axes[i, 3].imshow(gt_slice.T, cmap='Greens', alpha=0.4, origin='lower')
            axes[i, 3].imshow(pred_slice.T, cmap='Reds', alpha=0.4, origin='lower')
            axes[i, 3].set_title('Overlay (Green=GT, Red=Pred)')
            axes[i, 3].axis('off')

        plt.suptitle('3D View (Multiple Axial Slices)', fontsize=14, fontweight='bold')
        plt.tight_layout()

        view_output_path = output_path.replace('.png', '_3d.png')
        plt.savefig(view_output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def render_surface_mesh(self,
                           segmentation: np.ndarray,
                           spacing: tuple,
                           output_path: str,
                           color: str = 'red',
                           smooth_iterations: int = 20):
        """
        Generate smooth surface mesh using marching cubes.

        Args:
            segmentation: Binary mask (H, W, D)
            spacing: Voxel spacing in mm (sx, sy, sz)
            output_path: Path to save PNG and VTK file
            color: Mesh color
            smooth_iterations: Number of smoothing iterations
        """
        if not PYVISTA_AVAILABLE:
            print("PyVista not available. Skipping surface mesh rendering.")
            return

        # Create grid
        grid = pv.ImageData()
        grid.dimensions = np.array(segmentation.shape) + 1
        grid.spacing = spacing
        grid.point_data["values"] = segmentation.flatten(order="F")

        # Extract surface using marching cubes
        try:
            mesh = grid.contour([0.5], method='marching_cubes')

            # Smooth mesh
            if smooth_iterations > 0:
                mesh = mesh.smooth(n_iter=smooth_iterations, feature_smoothing=False,
                                  boundary_smoothing=True)

            # Render
            plotter = pv.Plotter(off_screen=True, window_size=self.window_size)
            plotter.add_mesh(mesh, color=color, smooth_shading=True)
            plotter.camera_position = [(1, 1, 1), (0, 0, 0), (0, 0, 1)]

            # Save screenshot
            plotter.screenshot(output_path.replace('.vtk', '.png'), transparent_background=True)
            plotter.close()

            # Save VTK file for external viewers
            vtk_path = output_path if output_path.endswith('.vtk') else output_path.replace('.png', '.vtk')
            mesh.save(vtk_path)

        except Exception as e:
            print(f"Warning: Surface extraction failed: {e}")

    def create_interactive_html(self,
                                ct_volume: np.ndarray,
                                segmentation: np.ndarray,
                                prediction: np.ndarray,
                                spacing: tuple,
                                output_path: str):
        """
        Create interactive HTML visualization using Plotly.

        Args:
            ct_volume: 3D CT data (H, W, D)
            segmentation: Ground truth binary mask (H, W, D)
            prediction: Prediction binary mask (H, W, D)
            spacing: Voxel spacing in mm
            output_path: Path to save HTML file
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Skipping interactive HTML generation.")
            return

        # Sample points for visualization (to avoid overwhelming browser)
        max_points = 10000

        # Ground truth points (green)
        gt_points = np.argwhere(segmentation > 0)
        if len(gt_points) > max_points:
            indices = np.random.choice(len(gt_points), max_points, replace=False)
            gt_points = gt_points[indices]
        gt_points_scaled = gt_points * np.array(spacing)

        # Prediction points (red)
        pred_points = np.argwhere(prediction > 0)
        if len(pred_points) > max_points:
            indices = np.random.choice(len(pred_points), max_points, replace=False)
            pred_points = pred_points[indices]
        pred_points_scaled = pred_points * np.array(spacing)

        # Create Plotly figure
        fig = go.Figure()

        # Add ground truth
        if len(gt_points_scaled) > 0:
            fig.add_trace(go.Scatter3d(
                x=gt_points_scaled[:, 0],
                y=gt_points_scaled[:, 1],
                z=gt_points_scaled[:, 2],
                mode='markers',
                marker=dict(size=2, color='green', opacity=0.6),
                name='Ground Truth'
            ))

        # Add prediction
        if len(pred_points_scaled) > 0:
            fig.add_trace(go.Scatter3d(
                x=pred_points_scaled[:, 0],
                y=pred_points_scaled[:, 1],
                z=pred_points_scaled[:, 2],
                mode='markers',
                marker=dict(size=2, color='red', opacity=0.6),
                name='Prediction'
            ))

        # Update layout
        fig.update_layout(
            title="Interactive 3D Segmentation Viewer",
            scene=dict(
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)",
                zaxis_title="Z (mm)",
                aspectmode='data'
            ),
            showlegend=True,
            width=1200,
            height=800
        )

        # Save HTML
        fig.write_html(output_path)

    def _render_fallback_slices(self,
                                ct_volume: np.ndarray,
                                segmentation: np.ndarray,
                                prediction: np.ndarray,
                                output_path: str):
        """
        Fallback to 2D slice visualization when PyVista is not available.

        Args:
            ct_volume: 3D CT data (H, W, D)
            segmentation: Ground truth binary mask (H, W, D)
            prediction: Prediction binary mask (H, W, D)
            output_path: Path to save PNG
        """
        H, W, D = ct_volume.shape
        mid_slice = D // 2

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # CT alone
        axes[0].imshow(ct_volume[:, :, mid_slice].T, cmap='gray', origin='lower')
        axes[0].set_title('CT Volume')
        axes[0].axis('off')

        # Ground truth overlay
        axes[1].imshow(ct_volume[:, :, mid_slice].T, cmap='gray', origin='lower')
        axes[1].imshow(segmentation[:, :, mid_slice].T, cmap='Greens', alpha=0.5, origin='lower')
        axes[1].set_title('Ground Truth Overlay')
        axes[1].axis('off')

        # Prediction overlay
        axes[2].imshow(ct_volume[:, :, mid_slice].T, cmap='gray', origin='lower')
        axes[2].imshow(prediction[:, :, mid_slice].T, cmap='Reds', alpha=0.5, origin='lower')
        axes[2].set_title('Prediction Overlay')
        axes[2].axis('off')

        # Both overlays
        axes[3].imshow(ct_volume[:, :, mid_slice].T, cmap='gray', origin='lower')
        axes[3].imshow(segmentation[:, :, mid_slice].T, cmap='Greens', alpha=0.4, origin='lower')
        axes[3].imshow(prediction[:, :, mid_slice].T, cmap='Reds', alpha=0.4, origin='lower')
        axes[3].set_title('Both Overlays\n(Green=GT, Red=Pred)')
        axes[3].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def create_comparison_panel(self,
                                ct_volume: np.ndarray,
                                segmentation: np.ndarray,
                                prediction: np.ndarray,
                                uncertainty_map: np.ndarray,
                                error_rgb: np.ndarray,
                                output_path: str):
        """
        Create comprehensive 5-panel comparison view.

        Args:
            ct_volume: 3D CT data (H, W, D)
            segmentation: Ground truth binary mask (H, W, D)
            prediction: Prediction binary mask (H, W, D)
            uncertainty_map: Uncertainty map (H, W, D)
            error_rgb: RGB error map (H, W, D, 3)
            output_path: Path to save PNG
        """
        # Use MIP for comprehensive view
        ct_mip = np.max(ct_volume, axis=2)
        gt_mip = np.max(segmentation, axis=2)
        pred_mip = np.max(prediction, axis=2)
        unc_mip = np.max(uncertainty_map, axis=2)
        err_mip = np.max(error_rgb, axis=2)

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))

        # CT volume
        axes[0].imshow(ct_mip.T, cmap='gray', origin='lower')
        axes[0].set_title('CT Volume\n(Axial MIP)', fontsize=12)
        axes[0].axis('off')

        # Ground truth
        axes[1].imshow(ct_mip.T, cmap='gray', origin='lower')
        axes[1].imshow(gt_mip.T, cmap='Greens', alpha=0.6, origin='lower')
        axes[1].set_title('Ground Truth Overlay\n(Green)', fontsize=12)
        axes[1].axis('off')

        # Prediction
        axes[2].imshow(ct_mip.T, cmap='gray', origin='lower')
        axes[2].imshow(pred_mip.T, cmap='Reds', alpha=0.6, origin='lower')
        axes[2].set_title('Prediction Overlay\n(Red)', fontsize=12)
        axes[2].axis('off')

        # Error overlay (err_mip is RGB, don't transpose - already (H, W, 3))
        axes[3].imshow(ct_mip.T, cmap='gray', origin='lower')
        axes[3].imshow(np.transpose(err_mip, (1, 0, 2)), alpha=0.7, origin='lower')
        axes[3].set_title('Error Overlay\n(Green=TP, Red=FP, Blue=FN)', fontsize=12)
        axes[3].axis('off')

        # Uncertainty map
        axes[4].imshow(ct_mip.T, cmap='gray', origin='lower')
        im = axes[4].imshow(unc_mip.T, cmap='hot', alpha=0.7, origin='lower')
        axes[4].set_title('Uncertainty Map\n(Hotter = More Uncertain)', fontsize=12)
        axes[4].axis('off')
        plt.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
