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
