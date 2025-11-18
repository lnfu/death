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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ._core import IRLSResult


@dataclass
class DEADataset:
    """
    Differential Expression Dataset.

    This class stores all intermediate and final results from the DEA pipeline.
    """

    # ==================== Input data ====================
    k_matrix: np.ndarray  # read counts (n_genes, n_samples)
    design_matrix: np.ndarray  # (n_samples, n_params)
    gene_names: Optional[np.ndarray] = None
    sample_names: Optional[np.ndarray] = None

    # ==================== Normalization results ====================
    sf: Optional[np.ndarray] = None  # size factors (n_samples,)
    # normalized counts (n_genes, n_samples)
    k_matrix_norm: Optional[np.ndarray] = None

    # ==================== Dispersion results ====================
    disp: Optional[np.ndarray] = None  # dispersion estimates (n_genes,)

    # ==================== Fitting results ====================
    beta: Optional[np.ndarray] = None  # coefficients (n_genes, n_params)
    converged: Optional[np.ndarray] = None  # convergence status (n_genes,)
    weights: Optional[np.ndarray] = None  # IRLS weights (n_genes, n_samples)
    se: Optional[np.ndarray] = None  # standard errors (n_genes, n_params)

    # ==================== Testing results ====================
    p_value: Optional[np.ndarray] = None  # (n_genes,)
    adj_p_value: Optional[np.ndarray] = None  # (n_genes,)
    log2_fc: Optional[np.ndarray] = None  # (n_genes,)
    wald_stat: Optional[np.ndarray] = None  # (n_genes,)

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if self.k_matrix.ndim != 2:
            raise ValueError("counts must be 2D array")
        if self.design_matrix.shape[0] != self.k_matrix.shape[1]:
            raise ValueError("design_matrix rows must match counts columns")

    @property
    def n_genes(self) -> int:
        return self.k_matrix.shape[0]

    @property
    def n_samples(self) -> int:
        return self.k_matrix.shape[1]

    @property
    def n_params(self) -> int:
        return self.design_matrix.shape[1]

    # TODO(Enfu)
    # @classmethod
    # def from_dataframe(
    #     cls, k_matrix_df: pd.DataFrame, design_matrix_df: pd.DataFrame
    # ):
    #     """
    #     Create from pandas DataFrame
    #     """
    #     return cls(
    #         k_matrix=k_matrix_df.values,
    #         design_matrix=design_matrix_df.values,
    #         gene_names=k_matrix_df.index.values,
    #         sample_names=k_matrix_df.columns.values,
    #     )

    def filter_genes(self, mask: np.ndarray) -> int:
        """
        Filter genes based on a boolean mask.

        This method ensures all gene-level data structures remain consistent
        after filtering.

        Parameters:
        -----------
        mask : np.ndarray
            Boolean array of shape (n_genes,). True = keep, False = remove

        Returns:
        --------
        n_filtered : int
            Number of genes removed
        """
        if len(mask) != self.n_genes:
            raise ValueError(
                f"Mask length ({len(mask)}) must match number of genes ({self.n_genes})"
            )

        n_filtered = np.sum(~mask)

        if n_filtered == 0:
            return 0

        if self.k_matrix is not None:
            self.k_matrix = self.k_matrix[mask, :]

        if self.gene_names is not None:
            self.gene_names = self.gene_names[mask]

        if self.k_matrix_norm is not None:
            self.k_matrix_norm = self.k_matrix_norm[mask, :]

        if self.disp is not None:
            self.disp = self.disp[mask]

        if self.beta is not None:
            self.beta = self.beta[mask, :]

        if self.converged is not None:
            self.converged = self.converged[mask]

        if self.weights is not None:
            self.weights = self.weights[mask, :]

        if self.se is not None:
            self.se = self.se[mask, :]

        if self.p_value is not None:
            self.p_value = self.p_value[mask]

        if self.adj_p_value is not None:
            self.adj_p_value = self.adj_p_value[mask]

        if self.log2_fc is not None:
            self.log2_fc = self.log2_fc[mask]

        if self.wald_stat is not None:
            self.wald_stat = self.wald_stat[mask]

        return n_filtered
