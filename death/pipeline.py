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
from typing import List, Optional

import numpy as np

from .dataset import DEADataset
from .result import DEAResult
from .steps.base import BaseStep
from .steps.dispersion import (
    DispersionStep,
    MomentDispersionStep,
    MedianDispersionStep,
    FixedDispersionStep,
)
from .steps.fitting import FittingStep, NegativeBinomialFittingStep
from .steps.normalization import NormalizationStep, RatioNormalizationStep
from .steps.testing import TestingStep, WaldTestingStep

logger = logging.getLogger(__name__)


class DEAPipeline:
    def __init__(self):
        self._normalization: Optional[NormalizationStep] = None
        self._dispersion: Optional[DispersionStep] = None
        self._fitting: Optional[FittingStep] = None
        self._testing: Optional[TestingStep] = None

    def set_normalization(self, step: NormalizationStep) -> "DEAPipeline":
        self._normalization = step
        return self

    def set_dispersion(self, step: DispersionStep) -> "DEAPipeline":
        self._dispersion = step
        return self

    def set_fitting(self, step: FittingStep) -> "DEAPipeline":
        self._fitting = step
        return self

    def set_testing(self, step: TestingStep) -> "DEAPipeline":
        self._testing = step
        return self

    def run(self, dataset: DEADataset) -> DEAResult:
        """
        Run the differential expression analysis pipeline on the given dataset.
        """
        self._validate_pipeline()

        steps: List[tuple[str, Optional[BaseStep]]] = [
            ("Normalization", self._normalization),
            ("Dispersion", self._dispersion),
            ("Fitting", self._fitting),
            ("Testing", self._testing),
        ]

        for step_name, step in steps:
            logger.info("")
            logger.info(f"Running {step_name}: {step}")
            step.run(dataset)

        return self._build_result(dataset)

    def _validate_pipeline(self):
        if any(
            s is None
            for s in [
                self._normalization,
                self._dispersion,
                self._fitting,
                self._testing,
            ]
        ):
            raise ValueError("All pipeline steps must be set")

    def _build_result(self, dataset: DEADataset) -> DEAResult:
        """
        Build a DEAResult from the processed dataset.

        Parameters:
        -----------
        dataset : DEADataset
            The dataset after running all pipeline steps

        Returns:
        --------
        result : DEAResult
            The final differential expression results
        """
        # Validate required fields
        required_fields = [
            "gene_names",
            "p_value",
            "adj_p_value",
            "log2_fc",
            "wald_stat",
        ]
        for field in required_fields:
            if getattr(dataset, field) is None:
                raise ValueError(
                    f"Dataset missing required field: {field}. "
                    "Make sure all pipeline steps completed successfully."
                )

        # Get gene names (use indices if not provided)
        gene_names = dataset.gene_names

        # Compute base mean (geometric mean of normalized counts)
        if dataset.k_matrix_norm is not None:
            # Use normalized counts for base mean
            base_mean = (
                np.exp(np.mean(np.log(dataset.k_matrix_norm + 1), axis=1)) - 1
            )
        else:
            # Fall back to raw counts
            base_mean = np.mean(dataset.k_matrix, axis=1)

        result = DEAResult(
            gene_names=gene_names,
            log2_fc=dataset.log2_fc,
            p_value=dataset.p_value,
            adj_p_value=dataset.adj_p_value,
            wald_stat=dataset.wald_stat,
            base_mean=base_mean,
        )

        logger.info(f"Pipeline complete: {result}")

        return result

    @classmethod
    def default(cls) -> "DEAPipeline":
        """
        Create a default DEAPipeline instance.
        """
        return (
            cls()
            .set_normalization(RatioNormalizationStep())
            .set_dispersion(MomentDispersionStep())
            .set_fitting(NegativeBinomialFittingStep())
            .set_testing(WaldTestingStep())
        )
