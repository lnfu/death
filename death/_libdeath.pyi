# _libdeath.pyi
from typing import overload, Tuple
from pathlib import Path

class CountMatrix:
    """High-performance gene expression count matrix for bioinformatics analysis."""

    def __init__(self, n_gene: int, n_sample: int) -> None:
        """
        Initialize a count matrix with specified dimensions.

        Parameters
        ----------
        n_gene : int
            Number of genes (rows)
        n_sample : int
            Number of samples (columns)
        """
        ...

    @property
    def n_gene(self) -> int:
        """Number of genes in the matrix"""
        ...

    @property
    def n_sample(self) -> int:
        """Number of samples in the matrix"""
        ...

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the matrix (n_gene, n_sample)"""
        ...

    @overload
    def __getitem__(self, index: Tuple[int, int]) -> float:
        """
        Get a value at the specified position.

        Parameters
        ----------
        index : Tuple[int, int]
            (gene_index, sample_index)

        Returns
        -------
        float
            The count value at the specified position
        """
        ...

    @overload
    def __setitem__(self, index: Tuple[int, int], value: float) -> None:
        """
        Set a value at the specified position.

        Parameters
        ----------
        index : Tuple[int, int]
            (gene_index, sample_index)
        value : float
            The value to set
        """
        ...

    def __len__(self) -> int:
        """Return the total number of elements (n_gene * n_sample)"""
        ...

    def __repr__(self) -> str:
        """Return a string representation of the CountMatrix"""
        ...

    def __str__(self) -> str:
        """Return a human-readable string representation of the CountMatrix"""
        ...

    @staticmethod
    @overload
    def from_file(filepath: str) -> "CountMatrix":
        """Load count matrix from file using string path"""
        ...

    @staticmethod
    @overload
    def from_file(filepath: Path) -> "CountMatrix":
        """Load count matrix from file using Path object"""
        ...

    @staticmethod
    @overload
    def from_file(filepath: bytes) -> "CountMatrix":
        """Load count matrix from file using bytes path"""
        ...

    @staticmethod
    def from_file(filepath: str | Path | bytes) -> "CountMatrix":
        """
        Load count matrix from file

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to the count matrix file
            Supported formats: .txt, .tsv

        Returns
        -------
        CountMatrix
            Loaded count matrix

        Examples
        --------
        >>> # Using string
        >>> matrix = CountMatrix.from_file("GSE229869_cell.cycle.rnaseq.counts.txt")
        >>>
        >>> # Using pathlib.Path
        >>> from pathlib import Path
        >>> matrix = CountMatrix.from_file(Path("GSE229869_cell.cycle.rnaseq.counts.txt"))

        Raises
        ------
        RuntimeError
            If file cannot be opened or has invalid format
        ValueError
            If filepath is not a string or Path-like object
        """
        ...
