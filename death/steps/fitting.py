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

from .._core import (
    IRLSSolver,
    IRLSSolverOptions,
    NegativeBinomialDistribution,
)
from ..dataset import DEADataset
from .base import BaseStep

logger = logging.getLogger(__name__)


class FittingStep(BaseStep):
    """
    Base class for fitting steps.
    """


class NegativeBinomialFittingStep(FittingStep):
    """
    Step to fit a Negative Binomial distribution to the given data.
    """

    def __init__(
        self,
        name: str = "negative_binomial_fitting",
        max_iter: int = 100,
        tol: float = 1e-6,
        n_threads: int = 4,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.max_iter = max_iter
        self.tol = tol
        self.n_threads = n_threads

    def run(self, dataset: DEADataset, **kwargs) -> None:
        """
        Run the Negative Binomial fitting step.
        """
        logger.info(
            f"Fitting Negative Binomial GLM (max_iter={self.max_iter}, tol={self.tol})"
        )

        # Create distribution
        nb_distribution = NegativeBinomialDistribution(dataset.disp)

        # Configure solver options
        opts = IRLSSolverOptions()
        opts.max_iter = self.max_iter
        opts.tol = self.tol
        opts.n_threads = self.n_threads

        # Create solver
        solver = IRLSSolver(
            dist=nb_distribution,
            X=dataset.design_matrix,
            sf=dataset.sf,
            n_genes=dataset.n_genes,
            n_samples=dataset.n_samples,
            n_params=dataset.n_params,
            opts=opts,
        )

        irls_result = solver.fit(dataset.k_matrix)

        dataset.beta = irls_result.beta.copy()
        dataset.converged = irls_result.converged.copy().astype(bool)
        dataset.weights = irls_result.weights.copy()
        dataset.se = irls_result.standard_errors.copy()

        logger.info(
            f"Fitting converged for {np.sum(dataset.converged)}/{dataset.n_genes} genes"
        )
