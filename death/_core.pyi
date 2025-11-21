"""
Core module for death package
"""

from __future__ import annotations
import numpy
import numpy.typing
import typing

__all__: list[str] = ["OLSOptions", "OLSResult", "OLSSolver"]

class OLSOptions:
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

class OLSResult:
    def __repr__(self) -> str: ...
    @property
    def coefficients(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def fitted_values(self) -> numpy.typing.NDArray[numpy.float64]: ...

class OLSSolver:
    def __init__(self) -> None:
        """
        Initialize OLS Solver
        """
    def solve(
        self,
        X: numpy.typing.NDArray[numpy.float64],
        Y: numpy.typing.NDArray[numpy.float64],
        opts: typing.Any = None,
    ) -> OLSResult:
        """
        Solve an OLS problem.

        Parameters
        ----------
        X : numpy.ndarray, shape (n, p)
            Design matrix (must be C-contiguous)
        Y : numpy.ndarray, shape (n, nrhs)
            Response matrix (must be C-contiguous)
        opts : OLSOptions, optional
            Solver options. If None, uses default options.

        Returns
        -------
        OLSResult
            Solving results with coefficients and fitted values
        """
