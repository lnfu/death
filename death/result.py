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
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .utils.filter import (
    filter_downregulated,
    filter_significant,
    filter_upregulated,
)
from .utils.visualization import plot_interactive_volcano, plot_volcano

logger = logging.getLogger(__name__)


@dataclass
class DEAResult:
    """Differential Expression Analysis Result container."""

    gene_names: np.ndarray
    log2_fc: np.ndarray
    p_value: np.ndarray
    adj_p_value: np.ndarray
    wald_stat: np.ndarray
    base_mean: Optional[np.ndarray] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame sorted by adj_p_value."""
        return self._create_dataframe()

    def significant(
        self, alpha: float = 0.05, log2fc_thres: float = 0.0
    ) -> pd.DataFrame:
        """Get significantly differentially expressed genes."""
        return filter_significant(self, alpha, log2fc_thres)

    def upregulated(
        self, alpha: float = 0.05, log2fc_thres: float = 1.0
    ) -> pd.DataFrame:
        """Get significantly upregulated genes."""
        return filter_upregulated(self, alpha, log2fc_thres)

    def downregulated(
        self, alpha: float = 0.05, log2fc_thres: float = -1.0
    ) -> pd.DataFrame:
        """Get significantly downregulated genes."""
        return filter_downregulated(self, alpha, log2fc_thres)

    def top_genes(
        self, n: int = 20, by: str = "adj_p_value", ascending: bool = True
    ) -> pd.DataFrame:
        """Get top N genes sorted by specified column."""
        df = self.to_dataframe()
        if by not in df.columns:
            raise ValueError(
                f"Invalid column: {by}. Must be one of {list(df.columns)}"
            )
        return df.sort_values(by, ascending=ascending).head(n)

    def summary(self, alpha: float = 0.05) -> None:
        """Print summary statistics."""
        n_total = len(self.gene_names)
        n_sig = np.sum(self.adj_p_value < alpha)
        n_up = np.sum((self.adj_p_value < alpha) & (self.log2_fc > 0))
        n_down = np.sum((self.adj_p_value < alpha) & (self.log2_fc < 0))

        logger.info("=" * 40)
        logger.info("Differential Expression Analysis Summary")
        logger.info("=" * 40)
        logger.info(f"Total genes tested: {n_total:,}")
        logger.info(
            f"Significant genes (FDR < {alpha}): {n_sig:,} ({100 * n_sig / n_total:.1f}%)"
        )
        logger.info(f"  - Upregulated: {n_up:,} ({100 * n_up / n_total:.1f}%)")
        logger.info(
            f"  - Downregulated: {n_down:,} ({100 * n_down / n_total:.1f}%)"
        )
        logger.info("=" * 40)

    def plot_volcano(self, **kwargs) -> Figure:
        return plot_volcano(
            self.log2_fc, self.adj_p_value, self.gene_names, **kwargs
        )

    def plot_interactive_volcano(
        self,
        alpha: float = 0.05,
        log2fc_thres: float = 1.0,
        output_path: Optional[str] = None,
    ):
        return plot_interactive_volcano(
            self.to_dataframe(), alpha, log2fc_thres, output_path
        )

    def _create_dataframe(
        self,
        mask: Optional[np.ndarray] = None,
        sort_by: str = "adj_p_value",
        ascending: bool = True,
    ) -> pd.DataFrame:
        """Create DataFrame from arrays with optional filtering."""
        indices = mask if mask is not None else slice(None)
        data = {
            "log2_fc": self.log2_fc[indices],
            "wald_stat": self.wald_stat[indices],
            "p_value": self.p_value[indices],
            "adj_p_value": self.adj_p_value[indices],
        }
        if self.base_mean is not None:
            data["base_mean"] = self.base_mean[indices]

        df = pd.DataFrame(data, index=self.gene_names[indices])
        return df.sort_values(sort_by, ascending=ascending)
