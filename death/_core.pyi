"""
IRLS Solver core module
"""
from __future__ import annotations
import numpy
import numpy.typing
import typing
__all__: list[str] = ['IRLSOptions', 'IRLSResult', 'IRLSSolver']
class IRLSOptions:
    def __init__(self) -> None:
        ...
    @property
    def max_iter(self) -> int:
        ...
    @max_iter.setter
    def max_iter(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def tol(self) -> float:
        ...
    @tol.setter
    def tol(self, arg0: typing.SupportsFloat) -> None:
        ...
class IRLSResult:
    @property
    def coefficients(self) -> list[float]:
        ...
    @property
    def converged(self) -> bool:
        ...
    @property
    def deviance(self) -> float:
        ...
    @property
    def fitted_values(self) -> list[float]:
        ...
    @property
    def iterations(self) -> int:
        ...
    @property
    def std_errors(self) -> list[float]:
        ...
class IRLSSolver:
    def __init__(self) -> None:
        """
        Initialize IRLS Solver
        """
    def fit(self, X: numpy.typing.NDArray[numpy.float64], y: numpy.typing.NDArray[numpy.float64], opts: IRLSOptions = ...) -> IRLSResult:
        """
        Fit an IRLS model.
                         
        Parameters
        ----------
        X : numpy.ndarray, shape (n, p)
            Design matrix
        y : numpy.ndarray, shape (n,)
            Response vector
        opts : IRLSOptions, optional
            Solver options
        
        Returns
        -------
        IRLSResult
            Fitting results
        """
__version__: str = '0.1.0'
