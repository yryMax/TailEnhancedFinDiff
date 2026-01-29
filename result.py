import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class EvalResult:
    """Single evaluation result for one generated dataset."""
    name: str
    shape: tuple
    metrics: Dict[str, Any] = field(default_factory=dict)
    data: np.ndarray = None  # Raw data for plotting

    def _format_value(self, key: str, value: Any) -> str:
        if isinstance(value, dict) and 'mean' in value:
            return f"{value['mean']:.2f}±{value['std']:.2f}"
        elif isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    def to_dict(self) -> Dict[str, str]:
        return {k: self._format_value(k, v) for k, v in self.metrics.items()}


@dataclass
class EvalResultCollection:
    """Collection of evaluation results for multiple generated datasets."""
    training_shape: tuple = None
    training_data: np.ndarray = None
    results: List[EvalResult] = field(default_factory=list)

    def append(self, result: EvalResult):
        self.results.append(result)

    def _format_value(self, value: Any, mean_prec: int = 2, std_prec: int = 2) -> str:
        if isinstance(value, dict) and 'mean' in value:
            return f"{value['mean']:.{mean_prec}f}±{value['std']:.{std_prec}f}"
        elif isinstance(value, (int, float, np.floating)):
            return f"{float(value):.{mean_prec}f}"
        return str(value)

    def _select_assets(self) -> Dict[str, int]:
        """Select assets with max, median, min std."""
        training_std = np.std(self.training_data, axis=0)
        sorted_indices = np.argsort(training_std)
        return {
            'Max Std': sorted_indices[-1],
            'Median Std': sorted_indices[len(sorted_indices) // 2],
            'Min Std': sorted_indices[0],
        }

    def to_console(self, metric_names: List[str] = None):
        """Print results to console."""
        if metric_names is None:
            metric_names = list(self.results[0].metrics.keys()) if self.results else []

        print(f"Training: {self.training_shape}")
        for r in self.results:
            print(f"{r.name}: {r.shape}")
        print("=" * 70)

        # Header
        header = ["Metric"] + [r.name for r in self.results]
        col_width = max(20, max(len(r.name) for r in self.results) + 2)
        print(f"{'Metric':<10}", end="")
        for r in self.results:
            print(f"{r.name:>{col_width}}", end="")
        print()
        print("-" * (10 + col_width * len(self.results)))

        for metric_name in metric_names:
            print(f"{metric_name:<10}", end="")
            for r in self.results:
                if metric_name in r.metrics:
                    print(f"{self._format_value(r.metrics[metric_name]):>{col_width}}", end="")
                else:
                    print(f"{'-':>{col_width}}", end="")
            print()

    def to_markdown(self, metric_names: List[str] = None) -> str:
        """Generate markdown table (rows=methods, cols=metrics)."""
        if metric_names is None:
            metric_names = list(self.results[0].metrics.keys()) if self.results else []

        lines = []
        header = ["Method"] + metric_names
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")

        for r in self.results:
            row = [r.name]
            for metric_name in metric_names:
                if metric_name in r.metrics:
                    row.append(self._format_value(r.metrics[metric_name], mean_prec=4, std_prec=2))
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def plot_distribution(self, save_path: Optional[str] = None, bins: int = 50):
        """
        Plot histograms for assets with max, median, min std.
        Layout: 3 rows (assets) × N cols (methods)
        Each subplot: method histogram + GT fitted normal curve
        """
        if self.training_data is None:
            raise ValueError("Training data not stored. Cannot plot.")

        asset_indices = self._select_assets()
        n_methods = len(self.results)
        n_assets = len(asset_indices)

        fig, axes = plt.subplots(n_assets, n_methods, figsize=(4 * n_methods, 3 * n_assets),
                                  sharex=True, sharey=True)
        if n_methods == 1:
            axes = axes.reshape(-1, 1)
        if n_assets == 1:
            axes = axes.reshape(1, -1)

        row_titles = {
            'Max Std': 'Daily Return Histogram of Max Std Asset',
            'Median Std': 'Daily Return Histogram of Median Std Asset',
            'Min Std': 'Daily Return Histogram of Min Std Asset',
        }

        # Compute global x range across all assets and methods
        all_data = []
        for asset_idx in asset_indices.values():
            all_data.append(self.training_data[:, asset_idx])
            for r in self.results:
                all_data.append(r.data[:, asset_idx])
        global_x_min = min(d.min() for d in all_data)
        global_x_max = max(d.max() for d in all_data)
        global_x_range = np.linspace(global_x_min, global_x_max, 200)

        for row_idx, (asset_label, asset_idx) in enumerate(asset_indices.items()):
            # GT data for this asset
            gt_data = self.training_data[:, asset_idx]
            gt_mean, gt_std = np.mean(gt_data), np.std(gt_data)

            for col_idx, result in enumerate(self.results):
                ax = axes[row_idx, col_idx]

                # Method's data for this asset
                method_data = result.data[:, asset_idx]

                # Plot histogram with global range
                ax.hist(method_data, bins=bins, density=True, alpha=0.7,
                        label=result.name, color='steelblue', edgecolor='white',
                        range=(global_x_min, global_x_max))

                # Plot GT fitted normal curve
                gt_pdf = stats.norm.pdf(global_x_range, gt_mean, gt_std)
                ax.plot(global_x_range, gt_pdf, 'r-', lw=2, label=f'GT (μ={gt_mean:.4f}, σ={gt_std:.4f})')

                # Labels
                if row_idx == 0:
                    ax.set_title(result.name)
                if col_idx == 0:
                    ax.set_ylabel('Frequency')

                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

        # Add row titles below each row
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4)  # Add space for row titles

        for row_idx, (asset_label, asset_idx) in enumerate(asset_indices.items()):
            row_title = row_titles.get(asset_label, f'Hist of {asset_label} Asset')
            ax_bottom = axes[row_idx, 0].get_position().y0
            fig.text(0.5, ax_bottom - 0.05, row_title, ha='center', va='top', fontsize=11, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        return fig
