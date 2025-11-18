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

import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from ..dataset import DEADataset
from .base import BaseStep

logger = logging.getLogger(__name__)


class TestingStep(BaseStep):
    """
    Base class for testing steps.
    """


class WaldTestingStep(TestingStep):
    """
    Step to perform Wald test for differential expression.
    """

    def __init__(
        self,
        name: str = "wald_testing",
        coef_index: int = 1,  # Which coefficient to test
        alternative: str = "two-sided",  # "two-sided", "greater", or "less"
        alpha: float = 0.05,  # Significance level for FDR correction
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.coef_index = coef_index
        self.alternative = alternative
        self.alpha = alpha

    def run(self, dataset: DEADataset, **kwargs) -> None:
        """
        Run the Wald test step.
        """
        logger.info(
            f"Running Wald test on coefficient {self.coef_index} ({self.alternative})"
        )

        n_filtered = dataset.filter_genes(mask=dataset.converged)
        logger.info(
            f"Filtered out {n_filtered} non-converged genes. Remaining: {dataset.n_genes}"
        )

        # Extract the coefficient of interest
        beta_test = dataset.beta[:, self.coef_index]
        se_test = dataset.se[:, self.coef_index]

        # Compute Wald statistics
        wald_stats = beta_test / se_test

        # Compute p-values based on alternative hypothesis
        p_values = self._compute_pvalues(wald_stats, self.alternative)

        # Multiple testing correction (Benjamini-Hochberg FDR)
        reject, adj_p_values, _, _ = multipletests(
            p_values, alpha=self.alpha, method="fdr_bh"
        )

        # Compute log2 fold change (for the tested coefficient)
        # For NB GLM with log link: log(μ) = X @ β
        # So β represents log fold change on natural log scale
        # To convert to log2: log2(FC) = log_e(FC) / log_e(2)
        log2_fc = beta_test / np.log(2)

        dataset.wald_stat = wald_stats
        dataset.p_value = p_values
        dataset.adj_p_value = adj_p_values
        dataset.log2_fc = log2_fc
        logger.info(
            f"Found {np.sum(reject)}/{dataset.n_genes} significant genes (FDR < {self.alpha})"
        )

    def _compute_pvalues(
        self, wald_stats: np.ndarray, alternative: str
    ) -> np.ndarray:
        """
        Compute p-values from Wald statistics.

        Parameters:
        -----------
        wald_stats : np.ndarray
            Wald statistics (beta / SE)
        alternative : str
            Type of test: "two-sided", "greater", or "less"

        Returns:
        --------
        p_values : np.ndarray
            P-values for each gene
        """
        if alternative == "two-sided":
            # H1: beta != 0
            p_values = 2 * norm.sf(np.abs(wald_stats))
        elif alternative == "greater":
            # H1: beta > 0 (upregulation)
            p_values = norm.sf(wald_stats)
        elif alternative == "less":
            # H1: beta < 0 (downregulation)
            p_values = norm.cdf(wald_stats)
        else:
            raise ValueError(
                f"Invalid alternative: {alternative}. "
                "Must be 'two-sided', 'greater', or 'less'"
            )

        return p_values
