#!/usr/bin/env python
"""
3D Visualization Evaluation Script

Command-line interface for generating comprehensive 3D visualizations
of trained segmentation models.

Usage:
    python scripts/evaluate_3d.py --experiment unet_baseline --top_k 5
    python scripts/evaluate_3d.py --experiment att_mamba2_unet --top_k 3 --data_dir /path/to/data
"""

import argparse
import os
import sys
import yaml

from src.visualization.evaluator import SegmentationEvaluator


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="3D Visualization for Coronary Artery Segmentation Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate unet_baseline experiment with top 5 best/worst samples
  python scripts/evaluate_3d.py --experiment unet_baseline --top_k 5

  # Evaluate with custom data directory
  python scripts/evaluate_3d.py --experiment att_mamba2_unet --top_k 3 --data_dir ./data/university

  # Evaluate on CPU
  python scripts/evaluate_3d.py --experiment unet_baseline --device cpu

  # Custom output directory
  python scripts/evaluate_3d.py --experiment unet_baseline --output_dir ./custom_viz
        """
    )

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name (must match a directory in results/)"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of best/worst samples to visualize (default: 5)"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override data directory from config (optional)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory (default: results/{experiment}/visualizations)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for inference (default: cuda)"
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of validation samples to process (default: all)"
    )

    return parser.parse_args()


def main():
    """Main evaluation pipeline."""
    args = parse_arguments()

    # Construct paths
    experiment_dir = os.path.join("results", args.experiment)

    if not os.path.exists(experiment_dir):
        print(f"Error: Experiment directory not found: {experiment_dir}")
        print(f"Available experiments:")
        results_dir = "results"
        if os.path.exists(results_dir):
            experiments = [d for d in os.listdir(results_dir)
                          if os.path.isdir(os.path.join(results_dir, d))]
            for exp in sorted(experiments):
                print(f"  - {exp}")
        sys.exit(1)

    config_path = os.path.join(experiment_dir, "config_snapshot.yaml")
    model_path = os.path.join(experiment_dir, "best_model.pth")

    # Check if required files exist
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found: {model_path}")
        sys.exit(1)

    # Load configuration
    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Determine data directory
    if args.data_dir is not None:
        data_dir = args.data_dir
        print(f"Using custom data directory: {data_dir}")
    else:
        data_dir = config.get("data", {}).get("data_dir", "./data/imagecas")
        print(f"Using data directory from config: {data_dir}")

    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    # Determine output directory
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(experiment_dir, "visualizations")

    print(f"Output directory: {output_dir}")

    # Print summary
    print("\n" + "="*70)
    print("3D VISUALIZATION EVALUATION")
    print("="*70)
    print(f"Experiment:       {args.experiment}")
    print(f"Model:            {config.get('model', {}).get('name', 'Unknown')}")
    print(f"Data directory:   {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Top K samples:    {args.top_k}")
    print(f"Device:           {args.device}")
    print("="*70 + "\n")

    # Initialize evaluator
    try:
        evaluator = SegmentationEvaluator(
            config=config,
            model_path=model_path,
            data_dir=data_dir,
            device=args.device,
            max_samples=args.max_samples
        )
    except Exception as e:
        print(f"Error initializing evaluator: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Run full evaluation
    try:
        metrics_df = evaluator.run_full_evaluation(
            output_dir=output_dir,
            top_k=args.top_k
        )

        # Print summary statistics
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        print(f"Total samples:    {len(metrics_df)}")
        print(f"Mean Dice:        {metrics_df['dice'].mean():.4f} ± {metrics_df['dice'].std():.4f}")
        print(f"Mean IoU:         {metrics_df['iou'].mean():.4f} ± {metrics_df['iou'].std():.4f}")
        print(f"Mean clDice:      {metrics_df['cldice'].mean():.4f} ± {metrics_df['cldice'].std():.4f}")
        print(f"Mean Hausdorff:   {metrics_df['hausdorff_95'].mean():.2f} ± {metrics_df['hausdorff_95'].std():.2f} mm")
        print("="*70 + "\n")

        print("Evaluation complete.")
        print(f"\nView results:")
        print(f"  HTML Report:    {os.path.join(output_dir, 'summary_report.html')}")
        print(f"  Metrics CSV:    {os.path.join(output_dir, 'metrics_per_sample.csv')}")
        print(f"  Visualizations: {os.path.join(output_dir, 'samples')}")

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
