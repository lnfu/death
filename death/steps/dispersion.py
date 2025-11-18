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
from ..dataset import DEADataset
from .base import BaseStep
import numpy as np

logger = logging.getLogger(__name__)


class DispersionStep(BaseStep):
    """
    Base class for dispersion calculation steps.
    """


class MomentDispersionStep(DispersionStep):
    """
    Step to calculate dispersion using the method of moments with shrinkage.
    """

    def __init__(
        self,
        name: str = "moment_dispersion",
        min_disp: float = 0.00001,
        max_disp: float = float("inf"),
        min_mean: float = 0.1,
        enable_shrinkage: bool = True,
        shrinkage_strength: float = 10.0,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.min_mean = min_mean
        self.use_shrinkage = enable_shrinkage
        self.shrinkage_strength = shrinkage_strength

    def run(self, dataset: DEADataset, **kwargs) -> None:
        """
        Run the dispersion calculation step with optional shrinkage.
        """
        # Calculate gene-wise dispersions
        mean_of_inverse_size_factors = np.mean(1.0 / dataset.sf)
        mean = np.mean(dataset.k_matrix_norm, axis=1)
        variance = np.var(dataset.k_matrix_norm, axis=1, ddof=1)

        disp_raw = (variance - mean * mean_of_inverse_size_factors) / (mean**2)

        # Filter genes
        valid_mask = (
            (mean > self.min_mean) & (disp_raw > 0) & np.isfinite(disp_raw)
        )

        n_filtered = np.sum(~valid_mask)
        logger.info(
            f"Filtering {n_filtered} genes due to: "
            f"low expression (mean < {self.min_mean}) or invalid dispersion"
        )

        # Apply shrinkage if enabled
        if self.use_shrinkage:
            disp_final = self._shrink_dispersions(
                disp_raw[valid_mask], mean[valid_mask]
            )
            logger.info("Applied dispersion shrinkage")
        else:
            disp_final = disp_raw[valid_mask]
            logger.info("Using raw dispersions (no shrinkage)")

        # Apply filter
        dataset.filter_genes(valid_mask)

        # Clip
        dataset.disp = np.clip(disp_final, self.min_disp, self.max_disp)

        # Log statistics
        self._log_dispersion_stats(dataset.disp)

    def _shrink_dispersions(
        self, disp_raw: np.ndarray, mean: np.ndarray
    ) -> np.ndarray:
        """
        Shrink dispersions towards a central value.

        Simple shrinkage strategy:
        - Calculate median dispersion as the target
        - Shrink each gene's dispersion towards the median
        - Genes with higher counts get less shrinkage (more reliable estimates)

        Parameters:
        -----------
        disp_raw : np.ndarray
            Raw dispersion estimates
        mean : np.ndarray
            Mean normalized counts for each gene

        Returns:
        --------
        disp_shrunk : np.ndarray
            Shrunk dispersion estimates
        """
        # Target: robust central estimate
        median_disp = np.median(disp_raw)

        # Shrinkage weight: higher count = higher weight on raw estimate
        # Formula: weight = sqrt(mean) / (sqrt(mean) + lambda)
        # lambda controls shrinkage strength (higher = more shrinkage)
        weights = np.sqrt(mean) / (np.sqrt(mean) + self.shrinkage_strength)

        # Shrink: weighted average of raw estimate and median
        disp_shrunk = weights * disp_raw + (1 - weights) * median_disp

        logger.info(f"  Shrinkage target (median): {median_disp:.6f}")
        logger.info(f"  Average shrinkage weight: {np.mean(weights):.3f}")

        return disp_shrunk

    def _log_dispersion_stats(self, disp: np.ndarray):
        logger.info("Dispersion statistics:")
        logger.info(f"  Min:    {np.min(disp):.6f}")
        logger.info(f"  Q1:     {np.percentile(disp, 25):.6f}")
        logger.info(f"  Median: {np.median(disp):.6f}")
        logger.info(f"  Mean:   {np.mean(disp):.6f}")
        logger.info(f"  Q3:     {np.percentile(disp, 75):.6f}")
        logger.info(f"  Max:    {np.max(disp):.6f}")


class MedianDispersionStep(DispersionStep):
    """
    Use median dispersion for all genes (simple but conservative).

    This is a quick fix that often works well in practice.
    """

    def __init__(
        self,
        name: str = "median_dispersion",
        min_mean: float = 0.1,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.min_mean = min_mean

    def run(self, dataset: DEADataset, **kwargs) -> None:
        """
        Calculate dispersions then use median for all genes.
        """
        # Calculate gene-wise dispersions
        mean_of_inverse_size_factors = np.mean(1.0 / dataset.sf)
        mean = np.mean(dataset.k_matrix_norm, axis=1)
        variance = np.var(dataset.k_matrix_norm, axis=1, ddof=1)

        disp_raw = (variance - mean * mean_of_inverse_size_factors) / (mean**2)

        # Filter
        valid_mask = (
            (mean > self.min_mean) & (disp_raw > 0) & np.isfinite(disp_raw)
        )

        n_filtered = np.sum(~valid_mask)
        logger.info(f"Filtering {n_filtered} genes")

        # Use median for ALL genes
        median_disp = np.median(disp_raw[valid_mask])

        dataset.filter_genes(valid_mask)
        dataset.disp = np.full(dataset.n_genes, median_disp)

        logger.info(f"Using median dispersion for all genes: {median_disp:.6f}")


class FixedDispersionStep(DispersionStep):
    """
    Use a fixed dispersion value for all genes (for testing/debugging).
    """

    def __init__(
        self,
        name: str = "fixed_dispersion",
        fixed_disp: float = 0.01,
        min_mean: float = 0.1,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.fixed_disp = fixed_disp
        self.min_mean = min_mean

    def run(self, dataset: DEADataset, **kwargs) -> None:
        """
        Use fixed dispersion for all genes.
        """
        mean = np.mean(dataset.k_matrix_norm, axis=1)
        valid_mask = mean > self.min_mean

        n_filtered = np.sum(~valid_mask)
        logger.info(f"Filtering {n_filtered} genes")

        dataset.filter_genes(valid_mask)
        dataset.disp = np.full(dataset.n_genes, self.fixed_disp)

        logger.info(
            f"Using fixed dispersion for all genes: {self.fixed_disp:.6f}"
        )
