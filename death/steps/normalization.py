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

from ..dataset import DEADataset
from .base import BaseStep

logger = logging.getLogger(__name__)


class NormalizationStep(BaseStep):
    """
    Base class for normalization steps.
    """


class RatioNormalizationStep(NormalizationStep):
    """
    Step to perform median of ratios normalization.
    """

    def __init__(self, name: str = "ratio_normalization", **kwargs):
        super().__init__(name, **kwargs)

    def run(self, dataset: DEADataset, **kwargs) -> None:
        """
        Run the ratio normalization step.
        """
        has_zeros = dataset.k_matrix == 0
        is_positive = dataset.k_matrix > 0

        log_k = np.empty_like(dataset.k_matrix, dtype=np.float64)
        log_k[~is_positive] = -np.inf
        log_k[is_positive] = np.log(dataset.k_matrix[is_positive])

        log_geo_means = np.mean(log_k, axis=1)

        valid_rows = ~np.any(has_zeros, axis=1)
        ratios = (
            np.log(dataset.k_matrix[valid_rows])
            - log_geo_means[valid_rows, np.newaxis]
        )
        if np.all(np.isnan(ratios)):
            raise ValueError("No valid genes found in k_matrix.")

        # Results - size factors
        dataset.sf = np.exp(np.median(ratios, axis=0))

        # Results - normalized k_matrix
        zero_rows = np.all(dataset.k_matrix == 0, axis=1)
        if np.any(zero_rows):
            logger.warning(
                f"Found {np.sum(zero_rows)} rows with all zeros."
                " They will be removed."
            )
            # dataset.k_matrix = dataset.k_matrix[~zero_rows, :]
            dataset.filter_genes(mask=~zero_rows)

        logger.info(f"Remaining {dataset.k_matrix.shape[0]} rows")
        dataset.k_matrix_norm = dataset.k_matrix / dataset.sf[np.newaxis, :]


class PoscountNormalizationStep(NormalizationStep):
    """
    Step to perform positive count normalization.
    """

    def __init__(self, name: str = "poscount_normalization", **kwargs):
        super().__init__(name, **kwargs)

    def run(self, dataset: DEADataset, **kwargs) -> None:
        """
        Run the positive count normalization step.
        """
        raise NotImplementedError(
            "PoscountNormalizationStep.run() is not implemented yet."
        )
