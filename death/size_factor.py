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

import numpy as np
from typing import Literal


def get_size_factors(
    k_matrix: np.ndarray,
    method: Literal[
        "ratio", "poscounts"
    ] = "ratio",  # TODO(Enfu): support "iterative" & more methods
):
    """
    Calculate size factors using the Median of Ratios method.

    Parameters
    ----------
    k_matrix : np.ndarray
        A 2D NumPy array where rows represent genes and columns represent
        samples.
    method : str, optional
        The method to use for size factor calculation. Options are "ratio" or
        "poscounts".
        Default is "ratio".

    Returns
    -------
    size_factors : np.ndarray
        A 1D NumPy array of size factors for each sample.

    Raises
    ------
    ValueError
        If k_matrix is not a 2D NumPy array, is empty, or if an unknown method
        is specified.
    """

    # check k_matrix validity
    if not isinstance(k_matrix, np.ndarray):
        raise ValueError("k_matrix must be a NumPy array.")
    if k_matrix.ndim != 2:
        raise ValueError("k_matrix must be a 2D array.")
    if k_matrix.size == 0:
        raise ValueError("k_matrix must not be empty.")

    k_matrix = k_matrix.astype(np.float64)

    has_zeros = k_matrix == 0
    is_positive = k_matrix > 0

    if method == "ratio":
        log_k = np.empty_like(k_matrix, dtype=np.float64)
        log_k[~is_positive] = -np.inf
        log_k[is_positive] = np.log(k_matrix[is_positive])
        log_geo_means = np.mean(log_k, axis=1)

        valid_rows = ~np.any(has_zeros, axis=1)
        ratios = (
            np.log(k_matrix[valid_rows]) - log_geo_means[valid_rows, np.newaxis]
        )
        if np.all(np.isnan(ratios)):
            raise ValueError("No valid genes found in k_matrix.")

        return np.exp(np.median(ratios, axis=0))

    elif method == "poscounts":
        log_k = np.zeros_like(k_matrix, dtype=np.float64)
        log_k[is_positive] = np.log(k_matrix[is_positive])
        log_geo_means = np.mean(log_k, axis=1)

        # If we set log(0) = 0 for zeros, this makes log(0) = log(1) = 0
        # The filtering mechanism uses logmeans > 0 as the criterion, which
        # would cause rows with only 0s and 1s in k_matrix to be considered
        # invalid (rather than just rows with only 0s)
        # Perhaps it would be better to determine row validity based on whether
        # the row has non-zero values in k_matrix?
        has_only_0_and_1 = np.all((k_matrix == 0) | (k_matrix == 1), axis=1)
        if np.all(has_only_0_and_1):
            raise ValueError("No valid genes found in k_matrix.")

        mask = ~(is_positive & ~has_only_0_and_1[:, np.newaxis])
        ratios = np.ma.array(log_k - log_geo_means[:, np.newaxis], mask=mask)

        median_ratios = np.ma.median(ratios, axis=0)
        size_factors = np.exp(median_ratios)

        # Normalize size factors to have geometric mean of 1
        return size_factors / np.exp(np.mean(np.log(size_factors)))

    else:
        raise ValueError(f"Unknown method: {method}")
