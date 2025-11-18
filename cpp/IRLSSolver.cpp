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
#include <pthread.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>

namespace death {

const std::vector<double>& IRLSResult::standard_errors() {
  if (!se_cache_) {
    // First call: compute and cache
    std::vector<double> se_all(n_genes_ * n_params_);

    for (int g = 0; g < n_genes_; ++g) {
      const double* w_g = weights_.data() + g * n_samples_;
      auto se_g = compute_se_from_weights(w_g);

      std::copy(se_g.begin(), se_g.end(), se_all.begin() + g * n_params_);
    }

    se_cache_ = std::move(se_all);
  }

  return *se_cache_;
}

const std::vector<double>& IRLSResult::covariance_matrices() {
  if (!cov_cache_) {
    std::vector<double> cov_all(n_genes_ * n_params_ * n_params_);

    for (int g = 0; g < n_genes_; ++g) {
      const double* w_g = weights_.data() + g * n_samples_;
      auto cov_g = compute_cov_from_weights(w_g);

      std::copy(cov_g.begin(), cov_g.end(),
                cov_all.begin() + g * n_params_ * n_params_);
    }

    cov_cache_ = std::move(cov_all);
  }

  return *cov_cache_;
}

std::vector<double> IRLSResult::standard_errors_single(int gene_idx) const {
  const double* w = weights_.data() + gene_idx * n_samples_;
  return compute_se_from_weights(w);
}

std::vector<double> IRLSResult::covariance_matrix_single(int gene_idx) const {
  const double* w = weights_.data() + gene_idx * n_samples_;
  return compute_cov_from_weights(w);
}

std::vector<double> IRLSResult::compute_se_from_weights(const double* w) const {
  // SE = sqrt(diag(Cov))
  auto cov = compute_cov_from_weights(w);

  std::vector<double> se(n_params_);
  for (int j = 0; j < n_params_; ++j) {
    double var_j = cov[j * n_params_ + j];
    se[j] = (var_j > 0) ? std::sqrt(var_j)
                        : std::numeric_limits<double>::quiet_NaN();
  }

  return se;
}

std::vector<double> IRLSResult::compute_cov_from_weights(
    const double* w) const {
  const int n = n_samples_;
  const int p = n_params_;

  std::vector<double> cov(p * p);

  // X^T W X (Fisher Information Matrix)
  std::vector<double> XtWX(p * p, 0.0);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < p; ++j) {
      for (int k = 0; k < p; ++k) {
        // XtWX[j,k] += X[i,j] * w[i] * X[i,k]
        XtWX[j * p + k] += X_[i * p + j] * w[i] * X_[i * p + k];
      }
    }
  }

  // (X^T W X)^{-1}
  // using Cholesky decomposition
  // Copy XtWX to cov (since LAPACK overwrites input)
  std::copy(XtWX.begin(), XtWX.end(), cov.begin());

  // Cholesky factorization (assuming positive definite)
  int info_chol = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', p, cov.data(), p);

  if (info_chol != 0) {
    // Cholesky failed = matrix is not positive definite
    // Fill with NaN
    std::fill(cov.begin(), cov.end(), std::numeric_limits<double>::quiet_NaN());
    return cov;
  }

  // inverse from Cholesky factorization
  int info_inv = LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'U', p, cov.data(), p);

  if (info_inv != 0) {
    // Inversion failed
    std::fill(cov.begin(), cov.end(), std::numeric_limits<double>::quiet_NaN());
    return cov;
  }

  // dpotri only fills upper triangle - copy to lower triangle for symmetry
  for (int i = 0; i < p; ++i) {
    for (int j = i + 1; j < p; ++j) {
      cov[j * p + i] = cov[i * p + j];
    }
  }

  return cov;
}

namespace {

struct ThreadContext {
  IRLSSolver* solver;
  const double* Y;
  double* beta_out;
  double* w_out;
  uint8_t* converged_out;
  std::atomic<int>* next_gene;
  int n_genes;
  int n_samples;
  int n_params;
};

}  // namespace

size_t workspace_size(int n_samples, int n_params) {
  const int n = n_samples;
  const int p = n_params;
  return n        // eta
         + n      // mu
         + n      // z
         + n      // w (working weights)
         + n * p  // WX
         + n      // Wz
         + p;     // beta_old
}

void* IRLSSolver::thread_worker(void* arg) {
  auto* ctx = static_cast<ThreadContext*>(arg);

  // preallocate workspace
  std::vector<double> workspace(workspace_size(ctx->n_samples, ctx->n_params));

  while (true) {
    int gene_idx = ctx->next_gene->fetch_add(1, std::memory_order_relaxed);

    if (gene_idx >= ctx->n_genes) break;

    const double* y_g = ctx->Y + gene_idx * ctx->n_samples;
    double* beta_g = ctx->beta_out + gene_idx * ctx->n_params;
    double* w_g = ctx->w_out + gene_idx * ctx->n_samples;

    bool converged =
        ctx->solver->fit_single(gene_idx, y_g, beta_g, w_g, workspace.data());
    ctx->converged_out[gene_idx] = converged;
  }

  return nullptr;
}

IRLSResult IRLSSolver::fit(const double* Y) {
  IRLSResult result(n_genes_, n_samples_, n_params_, X_);

  // If single-threaded or few genes, use single thread
  if (opts_.n_threads <= 1 || n_genes_ <= opts_.n_threads) {
    std::vector<double> workspace(workspace_size(n_samples_, n_params_));

    for (int g = 0; g < n_genes_; ++g) {
      const double* y_g = Y + g * n_samples_;
      double* beta_g = result.beta().data() + g * n_params_;
      double* w_g = result.weights().data() + g * n_samples_;

      result.converged()[g] = fit_single(g, y_g, beta_g, w_g, workspace.data());
    }

    return result;
  }

  // Multi-threaded processing
  std::atomic<int> next_gene(0);

  ThreadContext ctx{
      .solver = this,
      .Y = Y,
      .beta_out = result.beta().data(),
      .w_out = result.weights().data(),
      .converged_out = result.converged().data(),
      .next_gene = &next_gene,
      .n_genes = n_genes_,
      .n_samples = n_samples_,
      .n_params = n_params_,
  };

  // create threads
  std::vector<pthread_t> threads(opts_.n_threads);

  for (int t = 0; t < opts_.n_threads; ++t) {
    int rc = pthread_create(&threads[t], nullptr, thread_worker, &ctx);
    if (rc != 0) {
      // Thread creation failed, reduce the number of threads and continue
      threads.resize(t);
      break;
    }
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    pthread_join(thread, nullptr);
  }

  return result;
}

bool IRLSSolver::fit_single(int gene_idx, const double* y, double* beta_out,
                            double* w_out, double* workspace) {
  const int n = n_samples_;
  const int p = n_params_;

  // Workspace preparation
  double* eta = workspace;    // (n)
  double* mu = eta + n;       // (n)
  double* z = mu + n;         // (n)
  double* w = z + n;          // (n) - working weights
  double* WX = w + n;         // (n × p)
  double* Wz = WX + n * p;    // (n)
  double* beta_old = Wz + n;  // (p)

  // Initialize beta = 0
  std::fill(beta_out, beta_out + p, 0.0);

  // Initialize mu = y + 0.1
  // TODO(Enfu): Consider better initialization strategies
  // Different distributions might need different initializations
  for (int i = 0; i < n; ++i) {
    mu[i] = std::max(y[i] + 0.1, 0.1);
  }

  bool converged = false;

  // IRLS iteration
  for (int iter = 0; iter < opts_.max_iter; ++iter) {
    std::copy(beta_out, beta_out + p, beta_old);

    // Linear predictor: η = g(μ)
    for (int i = 0; i < n; ++i) {
      eta[i] = dist_->link(mu[i]);
    }

    // Working response: z = η + (y - μ) * g'(μ)
    for (int i = 0; i < n; ++i) {
      double g_prime = dist_->derivative_link(mu[i]);
      z[i] = eta[i] + (y[i] - mu[i]) * g_prime;
    }

    // Working weights: w = 1 / [ (g'(μ))² * V(μ) ]
    for (int i = 0; i < n; ++i) {
      double g_prime = dist_->derivative_link(mu[i]);
      double var = dist_->variance(mu[i], gene_idx);
      w[i] = 1.0 / (g_prime * g_prime * var + 1e-10);  // avoid div by zero
    }

    // Weighted Least Squares: min || sqrt(W)X @ beta - sqrt(W)z ||²
    // Compute sqrt(W) * X and sqrt(W) * z
    for (int i = 0; i < n; ++i) {
      double sqrt_w = std::sqrt(w[i]);
      Wz[i] = sqrt_w * z[i];
      for (int j = 0; j < p; ++j) {
        WX[i * p + j] = sqrt_w * X_[i * p + j];
      }
    }

    // Solve least squares using LAPACK dgels
    {
      // Make copies since LAPACKE_dgels overwrites the data
      std::vector<double> WX_copy(WX, WX + n * p);
      std::vector<double> Wz_copy(Wz, Wz + n);

      int info = LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', n, p, 1, WX_copy.data(),
                               p, Wz_copy.data(), 1);

      if (info != 0) {
        // Solve failed - keep old beta and return false
        std::copy(beta_old, beta_old + p, beta_out);
        // Still save the current weights (may be useful for diagnostics)
        std::copy(w, w + n, w_out);
        return false;
      }

      // Result is in the first p elements of Wz_copy
      std::copy(Wz_copy.begin(), Wz_copy.begin() + p, beta_out);
    }

    // Update μ = sf * g⁻¹(X @ β)
    for (int i = 0; i < n; ++i) {
      double xb = 0.0;
      for (int j = 0; j < p; ++j) {
        xb += X_[i * p + j] * beta_out[j];
      }
      double mu_new = sf_[i] * dist_->inverse_link(xb);
      mu[i] = std::max(mu_new, 0.01);  // ensure mu > 0
    }

    // Check convergence
    double max_delta = 0.0;
    for (int j = 0; j < p; ++j) {
      max_delta = std::max(max_delta, std::abs(beta_out[j] - beta_old[j]));
    }

    if (max_delta < opts_.tol) {
      converged = true;
      break;
    }
  }

  // Save final weights (for later SE/Cov computation)
  std::copy(w, w + n, w_out);

  return converged;
}

}  // namespace death