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
