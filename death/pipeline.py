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
from scipy.optimize import minimize

from ._core import OLSSolver
from .nb import nb_dnll, nb_nll
from .size_factor import get_size_factors


@dataclass
class PipelineConfig:
    min_disp: float = 1e-8
    max_disp: float = 10.0

    cox_reid_regularization: bool = True
    prior_regularization: bool = False


class Pipeline:
    def __init__(
        self, k_matrix: np.ndarray, config: PipelineConfig = PipelineConfig()
    ):
        self.k_matrix_raw = k_matrix
        self.config = config

        self.m = k_matrix.shape[0]  # number of genes
        self.n = k_matrix.shape[1]  # number of samples
        self.p = 5  # TODO(Enfu)

        # TODO(Enfu): write a function to generate design matrix
        self.design_matrix = np.array(
            [
                [1, 0, 0, 0, 0],  # T0_1
                [1, 0, 0, 0, 0],  # T0_2
                [1, 0, 0, 0, 0],  # T0_3
                [1, 1, 0, 0, 0],  # T1_1
                [1, 1, 0, 0, 0],  # T1_2
                [1, 1, 0, 0, 0],  # T1_3
                [1, 0, 1, 0, 0],  # T2_1
                [1, 0, 1, 0, 0],  # T2_2
                [1, 0, 1, 0, 0],  # T2_3
                [1, 0, 0, 1, 0],  # T3_1
                [1, 0, 0, 1, 0],  # T3_2
                [1, 0, 0, 1, 0],  # T3_3
                [1, 0, 0, 0, 1],  # T6_1
                [1, 0, 0, 0, 1],  # T6_2
                [1, 0, 0, 0, 1],  # T6_3
            ],
            dtype=np.float64,
        )

        self.sf: Optional[np.ndarray] = None
        self.k_matrix: Optional[np.ndarray] = None
        self.k_matrix_norm: Optional[np.ndarray] = None
        self.gene_wise_disp: Optional[np.ndarray] = None

    def run(self):
        # Calculate size factors
        self.sf = get_size_factors(self.k_matrix_raw)

        # Filter out rows with all zeros
        zero_rows = np.all(self.k_matrix_raw == 0, axis=1)
        if np.any(zero_rows):
            logging.warning(
                f"Found {np.sum(zero_rows)} rows with all zeros."
                " They will be removed."
            )
            self.k_matrix = self.k_matrix_raw[~zero_rows, :]
        else:
            self.k_matrix = self.k_matrix_raw

        self.k_matrix = np.ascontiguousarray(self.k_matrix.T)

        # Normalize k_matrix
        self.k_matrix_norm = self.k_matrix / self.sf[:, np.newaxis]

        # Gene-wise dispersion estimation
        self._gene_wise_dispersion_estimation()
        logging.info(f"Gene-wise dispersion estimates: {self.gene_wise_disp}")

    def _gene_wise_dispersion_estimation(self):
        logging.info("Estimating gene-wise dispersion...")

        # Fit OLS model for each gene to estimate dispersion
        solver = OLSSolver()
        result = solver.solve(
            self.design_matrix,
            self.k_matrix_norm,
        )
        k_matrix_norm_hat = result.fitted_values

        # TODO(Enfu): meaning of this?
        k_matrix_norm_hat_clipped = np.maximum(k_matrix_norm_hat, 1.0)

        disp_ols_fit = (
            (
                (self.k_matrix_norm - k_matrix_norm_hat_clipped) ** 2
                - k_matrix_norm_hat_clipped
            )
            / (k_matrix_norm_hat_clipped**2)
        ).sum(0) / (self.n - self.p)
        # TODO(Enfu): meaning of this?
        disp_ols_fit = np.maximum(disp_ols_fit, 0.0)

        # Use sample moments to estimate dispersion
        mean_of_inverse_size_factors = np.mean(1.0 / self.sf)
        mean = np.mean(self.k_matrix_norm, axis=0)
        variance = np.var(self.k_matrix_norm, axis=0, ddof=1)
        disp_sample_mom = (variance - mean * mean_of_inverse_size_factors) / (
            mean**2
        )

        # Combine two dispersion estimates by taking the minimum
        disp = np.minimum(disp_ols_fit, disp_sample_mom)

        # Clip dispersion estimates
        # TODO(Enfu): meaning of this?
        disp_hat = np.clip(disp, self.config.min_disp, self.config.max_disp)

        if True:
            # Important: no clipping here
            k_matrix_hat = k_matrix_norm_hat * self.sf[:, np.newaxis]
            k_matrix_hat = np.maximum(k_matrix_hat, 0.5)
        else:
            # TODO(Enfu): IRLS
            raise NotImplementedError()

        # Fitting gene-wise dispersion
        self.gene_wise_disp = np.zeros(self.k_matrix.shape[1], dtype=np.float64)
        for gene_idx in range(self.k_matrix.shape[1]):
            y = self.k_matrix[:, gene_idx]
            mu = k_matrix_hat[:, gene_idx]

            def loss(log_disp: float) -> float:
                disp = np.exp(log_disp)

                regular_term = 0.0

                if self.config.cox_reid_regularization:
                    W = mu / (1.0 + mu * disp)
                    # Fisher Information Matrix
                    regular_term += (
                        0.5
                        * np.linalg.slogdet(
                            (self.design_matrix.T * W) @ self.design_matrix
                        )[1]
                    )
                if self.config.prior_regularization:
                    raise NotImplementedError()

                return nb_nll(y, mu, disp) + regular_term

            def dloss(log_disp: float) -> float:
                disp = np.exp(log_disp)

                regular_term = 0.0

                if self.config.cox_reid_regularization:
                    W = mu / (1.0 + mu * disp)
                    dW = -(W**2)

                    regular_term += (
                        0.5
                        * np.trace(
                            np.linalg.inv(
                                (self.design_matrix.T * W) @ self.design_matrix
                            )
                            @ (self.design_matrix.T * dW)
                            @ self.design_matrix
                        )
                        * disp
                    )

                if self.config.prior_regularization:
                    raise NotImplementedError()

                return nb_dnll(y, mu, disp) * disp + regular_term

            result = minimize(
                lambda x: loss(x[0]),
                x0=np.asarray([np.log(disp_hat[gene_idx])]),
                jac=lambda x: np.asarray([dloss(x[0])]),
                method="L-BFGS-B",
                bounds=[
                    (np.log(self.config.min_disp), np.log(self.config.max_disp))
                ],
            )

            if not result.success:
                logging.warning(
                    f"Dispersion optimization for gene {gene_idx} did not "
                    "converge."
                )
            else:
                self.gene_wise_disp[gene_idx] = np.exp(result.x[0])
