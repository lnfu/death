/*
 * Copyright (c) 2025, Enfu Liao <efliao@cs.nycu.edu.tw>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * - Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "OLSSolver.hpp"

#include <cblas.h>
#include <lapacke.h>

#include <cstring>

OLSSolver::Result OLSSolver::solve(const double* X,  // design matrix
                                   const double* Y,  // response
                                   int n,            // number of samples
                                   int p,            // number of features
                                   int nrhs,  // number of right-hand sides
                                   const Options& opts) {
  // Solve OLS using LAPACK's DGELS
  std::vector<double> a(X, X + n * p);
  std::vector<double> b(Y, Y + n * nrhs);

  int lda = p;     // leading dimension of A
  int ldb = nrhs;  // leading dimension of B

  int info = LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', n, p, nrhs, a.data(), lda,
                           b.data(), ldb);

  if (info != 0) {
    throw std::runtime_error("LAPACKE_dgels failed with info = " +
                             std::to_string(info));
  }

  // Extract coefficients
  std::vector<double> coefficients(p * nrhs);
  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < nrhs; ++j) {
      coefficients[i * nrhs + j] = b[i * nrhs + j];
    }
  }

  // Compute fitted values
  std::vector<double> fitted_values(n * nrhs);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, nrhs,
              p,                            // 矩陣維度
              1.0,                          // alpha
              X, p,                         // A 和 lda
              coefficients.data(), nrhs,    // B 和 ldb
              0.0,                          // beta
              fitted_values.data(), nrhs);  // C 和 ldc

  return Result{nrhs, std::move(coefficients), std::move(fitted_values)};
}
