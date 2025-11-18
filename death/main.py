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
import pathlib
import re

import pandas as pd
import patsy
import typer
from rich.logging import RichHandler

import death


def main(
    count_filepath: str,
    output_dir: str = "outputs",
    alpha: float = 0.05,
    log2fc_thres: float = 1.0,
    label_top: int = 10,
    verbose: bool = True,
):
    """
    Run differential expression analysis and generate visualizations.

    Parameters:
    -----------
    count_filepath : str
        Path to the count matrix file (TSV format)
    output_prefix : str
        Prefix for output files (default: 'dea_results')
    alpha : float
        FDR threshold for significance (default: 0.05)
    log2fc_thres : float
        Log2 fold change threshold for volcano plot (default: 1.0)
    label_top : int
        Number of top genes to label in volcano plot (default: 10)
    verbose : bool
        Enable verbose logging (default: True)
    """
    # Setup logging
    if verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    logger = logging.getLogger(__name__)

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading count data from: {count_filepath}")
    df = pd.read_csv(count_filepath, sep="\t", header=0, index_col=0)
    logger.info(f"Loaded {df.shape[0]} genes × {df.shape[1]} samples")

    # Create design matrix
    conditions = pd.Categorical(
        [re.sub(r"\d+", "", col) for col in df.columns],
        categories=["untreated", "treated"],
    )
    design_matrix = patsy.dmatrix(
        "~ condition", pd.DataFrame({"condition": conditions})
    )

    k_matrix = df.to_numpy()

    result = death.run_dea(
        k_matrix, design_matrix, gene_names=df.index.to_numpy()
    )

    # Print summary
    result.summary(alpha=alpha)

    # Save full results
    full_results_path = output_dir / "full.csv"
    result.to_dataframe().to_csv(full_results_path)
    logger.info(f"Full results saved to: {full_results_path}")

    # Save upregulated genes
    up_path = output_dir / "upregulated.csv"
    result.upregulated(alpha=alpha, log2fc_thres=log2fc_thres).to_csv(up_path)
    logger.info(f"Upregulated genes saved to: {up_path}")

    # Save downregulated genes
    down_path = output_dir / "downregulated.csv"
    result.downregulated(alpha=alpha, log2fc_thres=-log2fc_thres).to_csv(
        down_path
    )
    logger.info(f"Downregulated genes saved to: {down_path}")

    # Generate volcano plot (PNG)
    result.plot_volcano(
        alpha=alpha,
        log2fc_thres=log2fc_thres,
        figsize=(12, 8),
        title="Differential Expression Analysis - Volcano Plot",
        label_top=label_top,
        save_path=output_dir / "volcano.png",  #  "volcano.pdf"
    )

    result.plot_interactive_volcano(
        alpha=alpha,
        log2fc_thres=log2fc_thres,
    ).show()


if __name__ == "__main__":
    typer.run(main)
