# Copyright (c) 2025, Enfu Liao <efliao@cs.nycu.edu.tw>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def plot_volcano(
    log2_fc: np.ndarray,
    adj_p_value: np.ndarray,
    gene_names: np.ndarray,
    alpha: float = 0.05,
    log2fc_thres: float = 1.0,
    figsize: tuple = (10, 8),
    title: str = "Volcano Plot",
    point_size: int = 20,
    label_top: int = 0,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Create a volcano plot of differential expression results.

    Parameters:
    -----------
    log2_fc : np.ndarray
        Log2 fold changes
    adj_p_value : np.ndarray
        Adjusted p-values
    gene_names : np.ndarray
        Gene names
    alpha : float
        FDR threshold for significance (default: 0.05)
    log2fc_thres : float
        Absolute log2 fold change threshold (default: 1.0)
    figsize : tuple
        Figure size (width, height) in inches
    title : str
        Plot title
    point_size : int
        Size of scatter points
    label_top : int
        Number of top genes to label (default: 0, no labels)
    save_path : str, optional
        Path to save the figure

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Calculate -log10(adjusted p-value)
    padj_safe = np.maximum(adj_p_value, 1e-300)
    neg_log10_padj = -np.log10(padj_safe)

    # Determine significance categories
    is_significant = adj_p_value < alpha
    is_upregulated = is_significant & (log2_fc > log2fc_thres)
    is_downregulated = is_significant & (log2_fc < -log2fc_thres)
    is_sig_not_fc = is_significant & (~is_upregulated) & (~is_downregulated)
    is_not_significant = ~is_significant

    # Count genes in each category
    n_up = np.sum(is_upregulated)
    n_down = np.sum(is_downregulated)
    n_sig_no_fc = np.sum(is_sig_not_fc)
    n_not_sig = np.sum(is_not_significant)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot points by category
    _plot_scatter_categories(
        ax,
        log2_fc,
        neg_log10_padj,
        is_not_significant,
        is_sig_not_fc,
        is_downregulated,
        is_upregulated,
        point_size,
        n_not_sig,
        n_sig_no_fc,
        n_down,
        n_up,
        log2fc_thres,
    )

    # Add threshold lines
    _add_threshold_lines(ax, alpha, log2fc_thres)

    # Label top genes if requested
    if label_top > 0:
        _label_top_genes(
            ax, gene_names, log2_fc, neg_log10_padj, adj_p_value, label_top
        )

    # Configure axes and styling
    _configure_plot_info(ax, title)

    plt.tight_layout()

    # Save and show
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Volcano plot saved to: {save_path}")

    return fig


def plot_interactive_volcano(
    df: pd.DataFrame,
    alpha: float = 0.05,
    log2fc_thres: float = 1.0,
    output_path: Optional[str] = None,
) -> plotly.graph_objs.Figure:
    """
    Create an interactive volcano plot using Plotly.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: log2_fc, adj_p_value, and optionally base_mean, p_value
    alpha : float
        FDR threshold for significance
    log2fc_thres : float
        Absolute log2 fold change threshold
    output_path : str, optional
        Path to save the HTML file

    Returns:
    --------
    fig : plotly.graph_objs.Figure
        The Plotly figure object
    """
    # Reset index to get gene names as a column
    plot_df = df.reset_index()
    plot_df = plot_df.rename(columns={plot_df.columns[0]: "gene"})

    # Classify genes
    plot_df["Significance"] = "Not Significant"
    plot_df.loc[
        (plot_df["adj_p_value"] < alpha) & (plot_df["log2_fc"] > log2fc_thres),
        "Significance",
    ] = "Upregulated"
    plot_df.loc[
        (plot_df["adj_p_value"] < alpha) & (plot_df["log2_fc"] < -log2fc_thres),
        "Significance",
    ] = "Downregulated"

    # Calculate -log10(padj)
    plot_df["-log10_padj"] = -np.log10(
        plot_df["adj_p_value"].clip(lower=1e-300)
    )

    # Create interactive plot
    fig = px.scatter(
        plot_df,
        x="log2_fc",
        y="-log10_padj",
        color="Significance",
        hover_data=["gene", "base_mean", "p_value", "adj_p_value"]
        if "base_mean" in plot_df.columns
        else ["gene", "p_value", "adj_p_value"],
        labels={
            "log2_fc": "Log2 Fold Change",
            "-log10_padj": "-Log10(Adjusted P-value)",
        },
        title="Interactive Volcano Plot",
        color_discrete_map={
            "Upregulated": "#d62728",
            "Downregulated": "#1f77b4",
            "Not Significant": "#d3d3d3",
        },
        opacity=0.7,
    )

    # Add threshold lines
    fig.add_hline(y=-np.log10(alpha), line_dash="dash", line_color="black")
    fig.add_vline(x=log2fc_thres, line_dash="dash", line_color="black")
    fig.add_vline(x=-log2fc_thres, line_dash="dash", line_color="black")

    # Update layout
    fig.update_layout(
        # width=1000,
        # height=700,
        hovermode="closest",
        template="plotly_white",
    )

    # Save if requested
    if output_path is not None:
        fig.write_html(output_path)
        logger.info(f"Interactive volcano plot saved to: {output_path}")

    return fig


def _plot_scatter_categories(
    ax,
    log2_fc,
    neg_log10_padj,
    is_not_significant,
    is_sig_not_fc,
    is_downregulated,
    is_upregulated,
    point_size,
    n_not_sig,
    n_sig_no_fc,
    n_down,
    n_up,
    log2fc_thres,
):
    """
    Plot scatter points for each significance category.
    """
    # Non-significant (gray)
    ax.scatter(
        log2_fc[is_not_significant],
        neg_log10_padj[is_not_significant],
        c="lightgray",
        s=point_size,
        alpha=0.5,
        label=f"Not significant (n={n_not_sig})",
        zorder=1,
    )

    # Significant but not passing FC threshold (cornflowerblue)
    if np.any(is_sig_not_fc):
        ax.scatter(
            log2_fc[is_sig_not_fc],
            neg_log10_padj[is_sig_not_fc],
            c="cornflowerblue",
            s=point_size,
            alpha=0.7,
            label=f"Sig., |log2FC| ≤ {log2fc_thres} (n={n_sig_no_fc})",
            zorder=2,
        )

    # Downregulated (blue)
    if np.any(is_downregulated):
        ax.scatter(
            log2_fc[is_downregulated],
            neg_log10_padj[is_downregulated],
            c="dodgerblue",
            s=point_size,
            alpha=0.8,
            label=f"Downregulated (n={n_down})",
            zorder=3,
        )

    # Upregulated (red)
    if np.any(is_upregulated):
        ax.scatter(
            log2_fc[is_upregulated],
            neg_log10_padj[is_upregulated],
            c="tomato",
            s=point_size,
            alpha=0.8,
            label=f"Upregulated (n={n_up})",
            zorder=3,
        )


def _add_threshold_lines(ax, alpha: float, log2fc_thres: float):
    """Add threshold lines to the plot."""
    neg_log10_alpha = -np.log10(alpha)
    ax.axhline(
        y=neg_log10_alpha,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=f"FDR = {alpha}",
    )

    if log2fc_thres > 0:
        ax.axvline(
            x=log2fc_thres,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )
        ax.axvline(
            x=-log2fc_thres,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label=f"|log2FC| = {log2fc_thres}",
        )


def _label_top_genes(
    ax, gene_names, log2_fc, neg_log10_padj, adj_p_value, label_top
):
    """Label the top N genes by adjusted p-value."""
    top_indices = np.argsort(adj_p_value)[:label_top]

    for idx in top_indices:
        ax.annotate(
            gene_names[idx],
            xy=(log2_fc[idx], neg_log10_padj[idx]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
            arrowprops=dict(
                arrowstyle="->", connectionstyle="arc3,rad=0", alpha=0.5
            ),
        )


def _configure_plot_info(ax, title: str):
    """
    Configure plot labels, title, grid, and legend
    """
    ax.set_xlabel("log₂ Fold Change", fontsize=12, fontweight="bold")
    ax.set_ylabel("-log₁₀ (Adjusted P-value)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    ax.legend(
        loc="upper right", frameon=True, fancybox=True, shadow=True, fontsize=10
    )
