"""
3D Visualization Package for Medical Image Segmentation

This package provides comprehensive 3D visualization capabilities for evaluating
coronary artery segmentation models, including volume rendering, error analysis,
uncertainty visualization, and interactive HTML reports.
"""

from .evaluator import SegmentationEvaluator
from .metrics_analyzer import compute_sample_metrics, create_metrics_table
from .volume_renderer import VolumeRenderer
from .error_visualization import compute_error_masks, render_error_overlay
from .uncertainty_maps import compute_entropy_map, compute_confidence_map
from .report_generator import generate_html_report

__all__ = [
    'SegmentationEvaluator',
    'compute_sample_metrics',
    'create_metrics_table',
    'VolumeRenderer',
    'compute_error_masks',
    'render_error_overlay',
    'compute_entropy_map',
    'compute_confidence_map',
    'generate_html_report',
]
