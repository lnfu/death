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

#include "IRLSSolver.hpp"

#include <lapacke.h>

#include <cmath>

IRLSSolver::Result IRLSSolver::fit(const double* X, int n, int p,
                                   const double* y,
                                   //    const double *offset,
                                   const Options& opts) {
  std::vector<double> eta(n), mu(n), z(n), w(n);
  std::vector<double> beta(p, 0.0);

  for (int iter = 0; iter < opts.max_iter; ++iter) {
    // linear predictor: η = g(μ)
    for (int i = 0; i < n; ++i) {
      eta[i] = dist_->link(mu[i]);
      // if (offset)
      //     eta[i] += offset[i];
    }

    // working response: z = η + (y - μ) * g'(μ)
    for (int i = 0; i < n; ++i) {
      double g_prime = dist_->derivative_link(mu[i]);
      z[i] = eta[i] + (y[i] - mu[i]) * g_prime;
    }

    // working weights: wi = 1 / [ (g'(μi))^2 * V(μi) ]
    for (int i = 0; i < n; ++i) {
      double var = dist_->variance(mu[i]);
      double g_prime = dist_->derivative_link(mu[i]);
      w[i] = 1.0 / (g_prime * g_prime * var);
    }

    // weighted least squares
    // β(i+1) = (X^T W X)^(-1) X^T W z
    // equivalently solve:
    // min ||WX * beta - Wz||^2

    std::vector<double> WX(n * p);
    std::vector<double> Wz(n);

    for (int i = 0; i < n; ++i) {
      double sqrt_w = std::sqrt(w[i]);
      Wz[i] = sqrt_w * z[i];
      for (int j = 0; j < p; ++j) {
        WX[i * p + j] = sqrt_w * X[i * p + j];
      }
    }

    // Solve the least squares problem using LAPACK's DGELS
    std::vector<double> beta_copy = beta;
    std::vector<double> WX_copy = WX;
    std::vector<double> Wz_copy = Wz;

    int32_t info = LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', n, p, 1, WX_copy.data(),
                                 p, Wz_copy.data(), 1);

    if (info != 0) {
      throw std::runtime_error("LAPACKE_dgels failed with info = " +
                               std::to_string(info));
    }

    beta = std::vector<double>(Wz_copy.begin(), Wz_copy.begin() + p);

    double max_delta = 0.0;
    for (int j = 0; j < p; ++j) {
      double delta = std::abs(beta[j] - beta_copy[j]);
      if (delta > max_delta) max_delta = delta;
    }

    if (max_delta < opts.tol) {
      return Result{beta, mu, {}, 0.0, iter + 1, true};
    }
  }
  return Result{beta, mu, {}, 0.0, opts.max_iter, false};
}
