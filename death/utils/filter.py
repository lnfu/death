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
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..result import DEAResult

logger = logging.getLogger(__name__)


def filter_significant(
    result: "DEAResult", alpha: float = 0.05, log2fc_thres: float = 0.0
) -> pd.DataFrame:
    """
    Get significantly differentially expressed genes.

    Parameters:
    -----------
    result : DEAResult
        The DEA result object
    alpha : float
        FDR threshold (default: 0.05)
    log2fc_thres : float
        Absolute log2 fold change threshold (default: 0.0)

    Returns:
    --------
    df : pd.DataFrame
        Significant genes sorted by adjusted p-value
    """
    mask = (result.adj_p_value < alpha) & (
        np.abs(result.log2_fc) > log2fc_thres
    )
    return _create_filtered_df(
        result, mask, "significant", alpha, log2fc_thres, "adj_p_value"
    )


def filter_upregulated(
    result: "DEAResult", alpha: float = 0.05, log2fc_thres: float = 1.0
) -> pd.DataFrame:
    """
    Get significantly upregulated genes.

    Parameters:
    -----------
    result : DEAResult
        The DEA result object
    alpha : float
        FDR threshold (default: 0.05)
    log2fc_thres : float
        Minimum log2 fold change (default: 1.0)

    Returns:
    --------
    df : pd.DataFrame
        Upregulated genes sorted by log2FC (descending)
    """
    mask = (result.adj_p_value < alpha) & (result.log2_fc > log2fc_thres)
    return _create_filtered_df(
        result,
        mask,
        "upregulated",
        alpha,
        log2fc_thres,
        "log2_fc",
        ascending=False,
    )


def filter_downregulated(
    result: "DEAResult", alpha: float = 0.05, log2fc_thres: float = -1.0
) -> pd.DataFrame:
    """
    Get significantly downregulated genes.

    Parameters:
    -----------
    result : DEAResult
        The DEA result object
    alpha : float
        FDR threshold (default: 0.05)
    log2fc_thres : float
        Maximum log2 fold change (default: -1.0)

    Returns:
    --------
    df : pd.DataFrame
        Downregulated genes sorted by log2FC (ascending)
    """
    mask = (result.adj_p_value < alpha) & (result.log2_fc < log2fc_thres)
    return _create_filtered_df(
        result,
        mask,
        "downregulated",
        alpha,
        log2fc_thres,
        "log2_fc",
        ascending=True,
    )


def _create_filtered_df(
    result: "DEAResult",
    mask: np.ndarray,
    category: str,
    alpha: float,
    threshold: float,
    sort_by: str,
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Create filtered DataFrame with logging.

    Parameters:
    -----------
    result : DEAResult
        The result object
    mask : np.ndarray
        Boolean mask for filtering
    category : str
        Category name for logging
    alpha : float
        FDR threshold (for logging)
    threshold : float
        Log2FC threshold (for logging)
    sort_by : str
        Column to sort by
    ascending : bool
        Sort order

    Returns:
    --------
    df : pd.DataFrame
        Filtered DataFrame or empty DataFrame if no genes match
    """
    if not np.any(mask):
        logger.info(
            f"No {category} genes found "
            f"(FDR < {alpha}, log2FC threshold = {threshold})"
        )
        return pd.DataFrame()

    return result._create_dataframe(mask, sort_by, ascending)
