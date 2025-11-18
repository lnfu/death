"""
DEATH: Differential Expression Analysis Tool for High-throughput NGS
"""

from __future__ import annotations
import numpy
import numpy.typing
import typing

__all__: list[str] = [
    "BernoulliDistribution",
    "Distribution",
    "IRLSResult",
    "IRLSSolver",
    "IRLSSolverOptions",
    "NegativeBinomialDistribution",
    "NormalDistribution",
    "PoissonDistribution",
]

class BernoulliDistribution(Distribution):
    def __init__(self) -> None:
        """
        Bernoulli distribution with logit link
        """

class Distribution:
    def derivative_link(self, mu: typing.SupportsFloat) -> float:
        """
        Derivative of link function g'(μ)
        """
    def inverse_link(self, eta: typing.SupportsFloat) -> float:
        """
        Inverse link function g⁻¹(η)
        """
    def link(self, mu: typing.SupportsFloat) -> float:
        """
        Link function g(μ)
        """
    def variance(
        self, mu: typing.SupportsFloat, idx: typing.SupportsInt
    ) -> float:
        """
        Variance function V(μ)
        """

class IRLSResult:
    """
    Results from IRLS fitting with lazy-evaluated statistics.

    This class stores the basic fitting results (beta, weights) and
    provides on-demand computation of statistical quantities (SE, Cov).
    Following the design pattern of statsmodels.GLMResults.
    """
    def __repr__(self) -> str: ...
    def covariance_matrix_single(
        self, gene_idx: typing.SupportsInt
    ) -> numpy.typing.NDArray[numpy.float64]:
        """
        Compute covariance matrix for a single gene (no caching).

        Parameters:
          gene_idx: Gene index

        Returns:
          numpy array of shape (n_params, n_params)
        """
    def standard_errors_single(
        self, gene_idx: typing.SupportsInt
    ) -> numpy.typing.NDArray[numpy.float64]:
        """
        Compute SE for a single gene (no caching).

        Parameters:
          gene_idx: Gene index

        Returns:
          numpy array of shape (n_params,)
        """
    @property
    def beta(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        Fitted coefficients (n_genes × n_params), zero-copy view
        """
    @property
    def converged(self) -> numpy.typing.NDArray[numpy.uint8]:
        """
        Convergence flags (n_genes), zero-copy view
        """
    @property
    def covariance_matrices(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        Covariance matrices (n_genes × n_params × n_params).
        Computed on first access and cached.
        Zero-copy view of cached result.
        """
    @property
    def n_genes(self) -> int:
        """
        Number of genes
        """
    @property
    def n_params(self) -> int:
        """
        Number of parameters
        """
    @property
    def n_samples(self) -> int:
        """
        Number of samples
        """
    @property
    def standard_errors(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        Standard errors (n_genes × n_params).
        Computed on first access and cached.
        Zero-copy view of cached result.
        """
    @property
    def weights(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        Final IRLS weights (n_genes × n_samples), zero-copy view
        """

class IRLSSolver:
    """
    Iteratively Reweighted Least Squares solver for GLMs.

    This class performs the numerical optimization (IRLS iteration).
    Statistical inference (SE, p-values) is handled by IRLSResult.
    """
    def __init__(
        self,
        dist: Distribution,
        X: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
        sf: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
        n_genes: typing.SupportsInt,
        n_samples: typing.SupportsInt,
        n_params: typing.SupportsInt,
        opts: typing.Any = None,
    ) -> None:
        """
        Initialize IRLS solver

        Parameters:
          dist: Distribution object
          X: Design matrix (n_samples × n_params)
          sf: Size factors (n_samples)
          n_genes: Number of genes
          n_samples: Number of samples
          n_params: Number of parameters
          opts: Solver options (optional, defaults to defaults())

        Example:
          >>> dist = NegativeBinomialDistribution(dispersions)
          >>> solver = IRLSSolver(dist, X, sf, 1000, 6, 2)
          >>> result = solver.fit(counts)
        """
    def fit(
        self, Y: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]
    ) -> IRLSResult:
        """
        Fit the GLM model for all genes.

        Parameters:
          Y: Count matrix (n_genes × n_samples)

        Returns:
          IRLSResult object with beta, weights, and lazy-evaluated statistics

        Example:
          >>> result = solver.fit(counts)
          >>> print(result.beta)  # Available immediately
          >>> print(result.standard_errors)  # Computed on first access
        """

class IRLSSolverOptions:
    @staticmethod
    def defaults() -> IRLSSolverOptions:
        """
        Get default options
        """
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, dict: dict) -> None:
        """
        Create options from dictionary
        """
    def __repr__(self) -> str: ...
    @property
    def max_iter(self) -> int:
        """
        Maximum number of iterations
        """
    @max_iter.setter
    def max_iter(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def n_threads(self) -> int:
        """
        Number of threads to use
        """
    @n_threads.setter
    def n_threads(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def tol(self) -> float:
        """
        Convergence tolerance
        """
    @tol.setter
    def tol(self, arg0: typing.SupportsFloat) -> None: ...

class NegativeBinomialDistribution(Distribution):
    def __init__(
        self,
        dispersions: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    ) -> None:
        """
        Negative binomial distribution with log link
        """

class NormalDistribution(Distribution):
    def __init__(self) -> None:
        """
        Normal distribution with identity link
        """

class PoissonDistribution(Distribution):
    def __init__(self) -> None:
        """
        Poisson distribution with log link
        """

__version__: str = "0.1.0"
