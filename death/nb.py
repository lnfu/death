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
from scipy.special import gammaln, polygamma


def nb_nll(y: np.ndarray, mu: np.ndarray, alpha: float) -> float:
    """
    Compute the negative log-likelihood of the Negative Binomial distribution.
    """
    # n = len(y)
    r = 1.0 / alpha

    first_term = gammaln(y + r) - gammaln(y + 1) - gammaln(r)

    second_term = r * (np.log(r) - np.log(mu + r))

    third_term = y * (np.log(mu) - np.log(mu + r))

    return -np.sum(first_term + second_term + third_term)


def nb_dnll(y: np.ndarray, mu: np.ndarray, alpha: float) -> float:
    """
    Compute the derivative of the negative log-likelihood of the Negative
    Binomial distribution.
    """
    r = 1.0 / alpha

    d_ll_d_r = np.sum(
        polygamma(0, y + r)
        - polygamma(0, r)
        + np.log(r)
        - np.log(mu + r)
        + 1
        - (y + r) / (mu + r)
    )

    # Chain Rule: d(NLL)/d(alpha) = - d(LL)/dr * dr/d(alpha)
    #                             = - d(LL)/dr * (-1/alpha^2)
    #                             = d(LL)/dr * (1/alpha^2)

    return d_ll_d_r * (1.0 / (alpha**2))
