"""
HTML Report Generation

This module generates comprehensive HTML evaluation reports with embedded
visualizations, interactive metrics tables, and summary statistics.
"""

import os
import pandas as pd
from .metrics_analyzer import get_sample_statistics


def generate_html_report(metrics_df: pd.DataFrame,
                        visualization_dir: str,
                        config: dict,
                        output_path: str,
                        selected_samples: list = None):
    """
    Generate comprehensive HTML evaluation report.

    Args:
        metrics_df: DataFrame with per-sample metrics
        visualization_dir: Directory containing visualization outputs
        config: Experiment configuration dictionary
        output_path: Path to save HTML report
        selected_samples: List of (category, rank, result_dict) tuples for selected samples
    """
    # Get summary statistics
    stats = get_sample_statistics(metrics_df)

    # Extract config information
    model_name = config.get('model', {}).get('name', 'Unknown')
    experiment_info = _format_experiment_info(config)

    # Build HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Segmentation Evaluation Report</title>
    <style>
        {_get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>3D Coronary Artery Segmentation Evaluation</h1>
            <p class="subtitle">Model: {model_name}</p>
        </header>

        <section id="summary">
            <h2>Executive Summary</h2>
            {_generate_summary_table(metrics_df, stats, experiment_info)}
        </section>

        <section id="metrics-dist">
            <h2>Metrics Distribution</h2>
            {_generate_metrics_distribution(metrics_df)}
        </section>

        <section id="best-performers">
            <h2>Best Performers</h2>
            {_generate_sample_gallery(metrics_df, visualization_dir, output_path, selected_samples, category='best')}
        </section>

        <section id="worst-performers">
            <h2>Worst Performers</h2>
            {_generate_sample_gallery(metrics_df, visualization_dir, output_path, selected_samples, category='worst')}
        </section>

        <section id="all-metrics">
            <h2>All Sample Metrics</h2>
            {_generate_full_metrics_table(metrics_df)}
        </section>

        <footer>
            <p>MIP Evaluation Framework - 3D Segmentation Analysis</p>
        </footer>
    </div>

    <script>
        {_get_javascript()}
    </script>
</body>
</html>
"""

    # Write HTML file
    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"HTML report saved to {output_path}")


def _get_css_styles() -> str:
    """Return CSS styles for the HTML report."""
    return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }

        header {
            text-align: center;
            padding: 30px 0;
            border-bottom: 3px solid #2c3e50;
            margin-bottom: 30px;
        }

        h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .subtitle {
            color: #7f8c8d;
            font-size: 1.2em;
        }

        h2 {
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db;
        }

        section {
            margin-bottom: 40px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }

        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }

        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
            cursor: pointer;
            user-select: none;
        }

        th:hover {
            background-color: #2980b9;
        }

        tr:hover {
            background-color: #f8f9fa;
        }

        .metric-value {
            font-weight: bold;
            color: #27ae60;
        }

        .sample-card {
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .sample-card h3 {
            color: #2c3e50;
            margin-bottom: 10px;
        }

        .sample-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin: 15px 0;
        }

        .metric-item {
            padding: 10px;
            background-color: #ecf0f1;
            border-radius: 4px;
        }

        .metric-label {
            font-size: 0.9em;
            color: #7f8c8d;
        }

        .metric-value-large {
            font-size: 1.3em;
            font-weight: bold;
            color: #2c3e50;
        }

        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }

        .image-item {
            text-align: center;
        }

        .image-item img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }

        .image-caption {
            margin-top: 8px;
            font-size: 0.9em;
            color: #7f8c8d;
        }

        footer {
            text-align: center;
            padding: 20px;
            margin-top: 40px;
            border-top: 2px solid #ecf0f1;
            color: #95a5a6;
        }

        .highlight-best {
            background-color: #d4edda;
        }

        .highlight-worst {
            background-color: #f8d7da;
        }

        .interactive-link {
            display: inline-block;
            padding: 8px 16px;
            background-color: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            margin: 5px;
        }

        .interactive-link:hover {
            background-color: #2980b9;
        }
    """


def _get_javascript() -> str:
    """Return JavaScript for interactive features."""
    return """
        // Make tables sortable
        function sortTable(table, col, reverse) {
            var tb = table.tBodies[0],
                tr = Array.prototype.slice.call(tb.rows, 0),
                i;
            reverse = -((+reverse) || -1);
            tr = tr.sort(function (a, b) {
                var aVal = a.cells[col].textContent.trim();
                var bVal = b.cells[col].textContent.trim();

                // Try to parse as number
                var aNum = parseFloat(aVal);
                var bNum = parseFloat(bVal);

                if (!isNaN(aNum) && !isNaN(bNum)) {
                    return reverse * (aNum - bNum);
                } else {
                    return reverse * aVal.localeCompare(bVal);
                }
            });
            for(i = 0; i < tr.length; ++i) tb.appendChild(tr[i]);
        }

        // Add click handlers to table headers
        document.addEventListener('DOMContentLoaded', function() {
            var tables = document.querySelectorAll('table');
            tables.forEach(function(table) {
                var ths = table.querySelectorAll('th');
                ths.forEach(function(th, index) {
                    th.addEventListener('click', function() {
                        sortTable(table, index, this.classList.contains('sorted'));

                        // Toggle sorted class
                        ths.forEach(function(h) { h.classList.remove('sorted'); });
                        this.classList.add('sorted');
                    });
                });
            });
        });
    """


def _format_experiment_info(config: dict) -> dict:
    """Extract and format experiment information."""
    info = {}

    # Model info
    model_config = config.get('model', {})
    info['model_name'] = model_config.get('name', 'Unknown')

    # Training info
    training_config = config.get('training', {})
    info['learning_rate'] = training_config.get('learning_rate', 'N/A')
    info['batch_size'] = training_config.get('batch_size', 'N/A')
    info['num_epochs'] = training_config.get('num_epochs', 'N/A')

    # Data info
    data_config = config.get('data', {})
    info['patch_size'] = data_config.get('patch_size', 'N/A')

    return info


def _generate_summary_table(metrics_df: pd.DataFrame, stats: dict, experiment_info: dict) -> str:
    """Generate HTML for summary statistics table."""
    n_samples = len(metrics_df)

    html = '<table>'
    html += '<tr><th>Property</th><th>Value</th></tr>'
    html += f'<tr><td>Model Architecture</td><td>{experiment_info.get("model_name", "N/A")}</td></tr>'
    html += f'<tr><td>Validation Samples</td><td>{n_samples}</td></tr>'
    html += '<tr><td colspan="2" style="background-color: #ecf0f1; font-weight: bold;">Performance Metrics</td></tr>'
    html += f'<tr><td>Mean Dice</td><td class="metric-value">{stats.get("dice_mean", 0):.4f} ± {stats.get("dice_std", 0):.4f}</td></tr>'
    html += f'<tr><td>Mean IoU</td><td class="metric-value">{stats.get("iou_mean", 0):.4f} ± {stats.get("iou_std", 0):.4f}</td></tr>'
    html += f'<tr><td>Mean clDice</td><td class="metric-value">{stats.get("cldice_mean", 0):.4f} ± {stats.get("cldice_std", 0):.4f}</td></tr>'
    html += f'<tr><td>Mean Smooth clDice</td><td class="metric-value">{stats.get("smooth_cldice_mean", 0):.4f} ± {stats.get("smooth_cldice_std", 0):.4f}</td></tr>'
    html += f'<tr><td>Mean Hausdorff (95%)</td><td class="metric-value">{stats.get("hausdorff_95_mean", 0):.2f} ± {stats.get("hausdorff_95_std", 0):.2f} mm</td></tr>'
    html += '<tr><td colspan="2" style="background-color: #ecf0f1; font-weight: bold;">Dice Score Range</td></tr>'
    html += f'<tr><td>Best (Max)</td><td class="metric-value">{stats.get("dice_max", 0):.4f}</td></tr>'
    html += f'<tr><td>Worst (Min)</td><td class="metric-value">{stats.get("dice_min", 0):.4f}</td></tr>'
    html += '</table>'

    return html


def _generate_metrics_distribution(metrics_df: pd.DataFrame) -> str:
    """Generate HTML for metrics distribution visualization."""
    # Simple text-based distribution for now
    # Could be enhanced with embedded Plotly charts
    html = '<div class="sample-metrics">'

    for metric in ['dice', 'iou', 'cldice', 'hausdorff_95']:
        if metric in metrics_df.columns:
            values = metrics_df[metric]
            html += '<div class="metric-item">'
            html += f'<div class="metric-label">{metric.upper()} Distribution</div>'
            html += f'<div class="metric-value-large">Range: [{values.min():.3f}, {values.max():.3f}]</div>'
            html += f'<div>Median: {values.median():.3f}</div>'
            html += f'<div>Q1: {values.quantile(0.25):.3f}, Q3: {values.quantile(0.75):.3f}</div>'
            html += '</div>'

    html += '</div>'
    return html


def _generate_sample_gallery(metrics_df: pd.DataFrame, visualization_dir: str,
                            output_path: str, selected_samples: list, category: str) -> str:
    """Generate HTML gallery for best/worst samples."""
    samples_dir = os.path.join(visualization_dir, "samples")

    # If selected_samples is provided, use it; otherwise fall back to old logic
    if selected_samples is not None:
        # Filter selected samples for this category
        category_samples = [(cat, rank, result) for cat, rank, result in selected_samples if cat == category]
        # Sort by rank
        category_samples = sorted(category_samples, key=lambda x: x[1])
    else:
        # Fallback to old logic (shouldn't happen with updated evaluator)
        if category == 'best':
            sample_df = metrics_df.head(5)
        else:
            sample_df = metrics_df.tail(5)
        category_samples = []
        for idx, (_, row) in enumerate(sample_df.iterrows(), 1):
            category_samples.append((category, idx, {'patient_id': row['patient_id']}))

    html = ''

    for cat, rank, result in category_samples:
        patient_id = result['patient_id']
        sample_name = f"{category}_{rank:03d}_{patient_id}"
        sample_dir = os.path.join(samples_dir, sample_name)

        # Get metrics for this sample from metrics_df
        # Find the row with matching patient_id and index (result contains index)
        if 'index' in result:
            row = metrics_df.iloc[result['index']]
        else:
            # Fallback: find by patient_id (may not be unique)
            row = metrics_df[metrics_df['patient_id'] == patient_id].iloc[0]

        html += '<div class="sample-card">'
        html += f'<h3>{category.capitalize()} #{rank}: {patient_id}</h3>'

        # Metrics
        html += '<div class="sample-metrics">'
        html += f'<div class="metric-item"><div class="metric-label">Dice</div><div class="metric-value-large">{row.get("dice", 0):.4f}</div></div>'
        html += f'<div class="metric-item"><div class="metric-label">IoU</div><div class="metric-value-large">{row.get("iou", 0):.4f}</div></div>'
        html += f'<div class="metric-item"><div class="metric-label">clDice</div><div class="metric-value-large">{row.get("cldice", 0):.4f}</div></div>'
        html += f'<div class="metric-item"><div class="metric-label">Hausdorff</div><div class="metric-value-large">{row.get("hausdorff_95", 0):.2f} mm</div></div>'
        html += '</div>'

        # Images
        html += '<div class="image-grid">'

        # Check which images exist
        images = [
            ('volume_render_axial.png', 'Axial View'),
            ('volume_render_3d.png', '3D View'),
            ('error_overlay.png', 'Error Overlay'),
            ('uncertainty_map.png', 'Uncertainty Map'),
            ('comparison_panel.png', 'Comparison Panel'),
        ]

        for img_file, caption in images:
            img_path = os.path.join(sample_dir, img_file)
            rel_path = os.path.relpath(img_path, os.path.dirname(output_path))

            if os.path.exists(img_path):
                html += '<div class="image-item">'
                html += f'<img src="{rel_path}" alt="{caption}">'
                html += f'<div class="image-caption">{caption}</div>'
                html += '</div>'

        html += '</div>'

        # Interactive link
        interactive_path = os.path.join(sample_dir, "interactive.html")
        if os.path.exists(interactive_path):
            rel_path = os.path.relpath(interactive_path, os.path.dirname(output_path))
            html += f'<a href="{rel_path}" class="interactive-link" target="_blank">View Interactive 3D</a>'

        html += '</div>'

    return html


def _generate_full_metrics_table(metrics_df: pd.DataFrame) -> str:
    """Generate HTML table with all sample metrics."""
    html = '<table id="metrics-table">'
    html += '<thead><tr>'

    # Column headers
    for col in metrics_df.columns:
        html += f'<th>{col}</th>'

    html += '</tr></thead>'
    html += '<tbody>'

    # Data rows
    for _, row in metrics_df.iterrows():
        html += '<tr>'
        for col in metrics_df.columns:
            value = row[col]
            if isinstance(value, float):
                html += f'<td>{value:.4f}</td>'
            else:
                html += f'<td>{value}</td>'
        html += '</tr>'

    html += '</tbody></table>'

    return html
