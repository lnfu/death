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
import pytest

from death.size_factor import get_size_factors


@pytest.fixture(scope="module")
def k_matrix_all_zeros():
    """Matrix with all zero values."""
    return np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])


@pytest.fixture(scope="module")
def k_matrix_all_zeros_and_ones():
    """Matrix with rows of all zeros and all ones."""
    return np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )


@pytest.fixture(scope="module")
def k_matrix_all_positive():
    """Matrix with all positive count values."""
    return np.array(
        [
            [31, 39, 53, 31, 52, 50, 26, 32, 29, 30, 38, 28, 24, 8, 41],
            [6, 13, 5, 5, 3, 14, 4, 49, 24, 1, 6, 12, 5, 5, 16],
            [3, 7, 8, 5, 13, 11, 1, 8, 3, 6, 2, 11, 9, 1, 8],
            [
                36,
                200,
                83,
                59,
                160,
                254,
                88,
                360,
                105,
                78,
                258,
                233,
                192,
                41,
                213,
            ],
            [42, 75, 23, 30, 75, 68, 74, 108, 25, 24, 89, 66, 75, 4, 58],
            [
                410,
                1426,
                400,
                416,
                977,
                1618,
                817,
                2547,
                561,
                342,
                2259,
                990,
                1313,
                299,
                1452,
            ],
        ]
    )


@pytest.fixture(scope="module")
def k_matrix_with_some_zeros():
    """Matrix with some zero values but all rows have non-zeros."""
    return np.array(
        [
            [31, 39, 53, 31, 52, 50, 26, 32, 29, 30, 38, 28, 24, 8, 41],
            [6, 13, 5, 5, 3, 14, 4, 49, 24, 1, 6, 12, 5, 5, 16],
            [3, 7, 8, 5, 13, 11, 1, 8, 3, 6, 2, 11, 9, 1, 8],
            [0, 0, 0, 0, 0, 1, 3, 2, 0, 0, 2, 6, 4, 0, 1],
            [
                36,
                200,
                83,
                59,
                160,
                254,
                88,
                360,
                105,
                78,
                258,
                233,
                192,
                41,
                213,
            ],
            [42, 75, 23, 30, 75, 68, 74, 108, 25, 24, 89, 66, 75, 4, 58],
            [
                410,
                1426,
                400,
                416,
                977,
                1618,
                817,
                2547,
                561,
                342,
                2259,
                990,
                1313,
                299,
                1452,
            ],
            [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 2, 0, 0, 0, 1],
        ]
    )


@pytest.fixture(scope="module")
def k_matrix_with_all_zero_rows():
    """Matrix with some rows containing all zeros."""
    return np.array(
        [
            [31, 39, 53, 31, 52, 50, 26, 32, 29, 30, 38, 28, 24, 8, 41],
            [6, 13, 5, 5, 3, 14, 4, 49, 24, 1, 6, 12, 5, 5, 16],
            [3, 7, 8, 5, 13, 11, 1, 8, 3, 6, 2, 11, 9, 1, 8],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 3, 2, 0, 0, 2, 6, 4, 0, 1],
            [
                36,
                200,
                83,
                59,
                160,
                254,
                88,
                360,
                105,
                78,
                258,
                233,
                192,
                41,
                213,
            ],
            [42, 75, 23, 30, 75, 68, 74, 108, 25, 24, 89, 66, 75, 4, 58],
            [
                410,
                1426,
                400,
                416,
                977,
                1618,
                817,
                2547,
                561,
                342,
                2259,
                990,
                1313,
                299,
                1452,
            ],
            [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 2, 0, 0, 0, 1],
        ]
    )


@pytest.fixture(scope="module")
def k_matrix_with_all_zero_and_one_rows():
    """Matrix with rows of all zeros and rows with single ones."""
    return np.array(
        [
            [31, 39, 53, 31, 52, 50, 26, 32, 29, 30, 38, 28, 24, 8, 41],
            [6, 13, 5, 5, 3, 14, 4, 49, 24, 1, 6, 12, 5, 5, 16],
            [3, 7, 8, 5, 13, 11, 1, 8, 3, 6, 2, 11, 9, 1, 8],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 3, 2, 0, 0, 2, 6, 4, 0, 1],
            [
                36,
                200,
                83,
                59,
                160,
                254,
                88,
                360,
                105,
                78,
                258,
                233,
                192,
                41,
                213,
            ],
            [42, 75, 23, 30, 75, 68, 74, 108, 25, 24, 89, 66, 75, 4, 58],
            [
                410,
                1426,
                400,
                416,
                977,
                1618,
                817,
                2547,
                561,
                342,
                2259,
                990,
                1313,
                299,
                1452,
            ],
            [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 2, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        ]
    )


@pytest.fixture(scope="module")
def expected_size_factors_standard():
    """Standard expected size factors for valid matrices."""
    return np.array(
        [
            0.69963849,
            1.62587611,
            0.66332811,
            0.67407287,
            1.43796149,
            1.90368182,
            0.75407337,
            2.61761004,
            0.74045346,
            0.57437396,
            1.54963444,
            1.54890153,
            1.53180917,
            0.28551078,
            1.6419232,
        ]
    )


@pytest.fixture(scope="module")
def expected_size_factors_poscounts():
    """Expected size factors for poscounts method."""
    return np.array(
        [
            0.66772077,
            1.55170314,
            0.63306688,
            0.64332146,
            1.37236125,
            1.81683527,
            0.71967231,
            2.49819386,
            0.70667374,
            0.54817084,
            1.47893964,
            1.47824017,
            1.46192756,
            0.27248569,
            1.56701816,
        ]
    )


@pytest.fixture(scope="module")
def expected_size_factors_poscounts_with_zeros():
    """Expected size factors for poscounts method with some zeros."""
    return np.array(
        [
            0.66842923,
            1.55334952,
            0.64487494,
            0.64400403,
            1.37381735,
            1.6520978,
            0.84679288,
            2.31528352,
            0.70742354,
            0.54875246,
            1.54595786,
            1.54769985,
            1.48663682,
            0.27277481,
            1.3836162,
        ]
    )


class TestInputValidation:
    """Test input validation for get_size_factors function."""

    def test_raises_error_on_non_ndarray(self):
        """Should raise ValueError when input is not a NumPy array."""
        with pytest.raises(ValueError, match="k_matrix must be a NumPy array."):
            get_size_factors(k_matrix=[[1, 2], [3, 4]])

    def test_raises_error_on_non_2d_array(self):
        """Should raise ValueError when input is not 2D."""
        with pytest.raises(ValueError, match="k_matrix must be a 2D array."):
            get_size_factors(k_matrix=np.array([1, 2, 3, 4, 5, 6]))

    def test_raises_error_on_empty_array(self):
        """Should raise ValueError when input is empty."""
        with pytest.raises(ValueError, match="k_matrix must not be empty."):
            get_size_factors(k_matrix=np.array([[], []]))

    def test_raises_error_on_unknown_method(self):
        """Should raise ValueError for unknown normalization method."""
        k_matrix = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="Unknown method"):
            get_size_factors(k_matrix=k_matrix, method="unknown_method")


class TestRatioMethod:
    """Test size factor calculation using ratio method."""

    def test_raises_error_with_all_zeros(self, k_matrix_all_zeros):
        """Should raise ValueError when all values are zero."""
        with pytest.raises(ValueError):
            get_size_factors(k_matrix=k_matrix_all_zeros, method="ratio")

    def test_with_all_zeros_and_ones(self, k_matrix_all_zeros_and_ones):
        """Should return uniform size factors for all zeros and ones."""
        expected = np.ones(15)
        result = get_size_factors(
            k_matrix=k_matrix_all_zeros_and_ones, method="ratio"
        )
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "k_matrix_all_positive",
            "k_matrix_with_some_zeros",
            "k_matrix_with_all_zero_rows",
            "k_matrix_with_all_zero_and_one_rows",
        ],
    )
    def test_with_various_matrices(
        self, fixture_name, expected_size_factors_standard, request
    ):
        """Should calculate consistent size factors for various valid matrices."""
        k_matrix = request.getfixturevalue(fixture_name)
        result = get_size_factors(k_matrix=k_matrix, method="ratio")
        np.testing.assert_allclose(
            result, expected_size_factors_standard, rtol=1e-6
        )


class TestPoscountsMethod:
    """Test size factor calculation using poscounts method."""

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "k_matrix_all_zeros",
            "k_matrix_all_zeros_and_ones",
        ],
    )
    def test_raises_error_with_invalid_matrices(self, fixture_name, request):
        """Should raise ValueError for matrices with all zeros or only zeros/ones."""
        k_matrix = request.getfixturevalue(fixture_name)
        with pytest.raises(ValueError):
            get_size_factors(k_matrix=k_matrix, method="poscounts")

    def test_with_all_positive_counts(
        self, k_matrix_all_positive, expected_size_factors_poscounts
    ):
        """Should calculate correct size factors for all positive counts."""
        result = get_size_factors(
            k_matrix=k_matrix_all_positive, method="poscounts"
        )
        np.testing.assert_allclose(
            result, expected_size_factors_poscounts, rtol=1e-6
        )

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "k_matrix_with_some_zeros",
            "k_matrix_with_all_zero_rows",
            "k_matrix_with_all_zero_and_one_rows",
        ],
    )
    def test_with_matrices_containing_zeros(
        self, fixture_name, expected_size_factors_poscounts_with_zeros, request
    ):
        """Should calculate consistent size factors for matrices with zeros."""
        k_matrix = request.getfixturevalue(fixture_name)
        result = get_size_factors(k_matrix=k_matrix, method="poscounts")
        np.testing.assert_allclose(
            result, expected_size_factors_poscounts_with_zeros, rtol=1e-6
        )
