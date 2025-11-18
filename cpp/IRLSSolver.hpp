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

#ifndef IRLSSOLVER_HPP
#define IRLSSOLVER_HPP

#include <memory>
#include <optional>
#include <vector>

#include "Distribution.hpp"

namespace death {

/**
 * @brief Result container for IRLS fitting with lazy-evaluated statistics
 */
class IRLSResult {
 public:
  IRLSResult(int n_genes, int n_samples, int n_params, const double* X)
      : n_genes_(n_genes),
        n_samples_(n_samples),
        n_params_(n_params),
        X_(X),
        beta_(n_genes * n_params),
        weights_(n_genes * n_samples),
        converged_(n_genes) {}

  int n_genes() const { return n_genes_; }
  int n_samples() const { return n_samples_; }
  int n_params() const { return n_params_; }

  std::vector<double>& beta() { return beta_; }
  const std::vector<double>& beta() const { return beta_; }

  std::vector<double>& weights() { return weights_; }
  const std::vector<double>& weights() const { return weights_; }

  std::vector<uint8_t>& converged() { return converged_; }
  const std::vector<uint8_t>& converged() const { return converged_; }

  /**
   * @brief Compute standard errors (SE) for all genes
   * @return Vector of SE (n_genes × n_params), computed on first call and
   * cached
   */
  const std::vector<double>& standard_errors();

  /**
   * @brief Compute covariance matrices for all genes
   * @return Vector of covariance matrices (n_genes × n_params × n_params)
   */
  const std::vector<double>& covariance_matrices();

  /**
   * @brief Compute SE for a single gene (without caching)
   * @param gene_idx Gene index
   * @return Vector of SE (n_params)
   */
  std::vector<double> standard_errors_single(int gene_idx) const;

  /**
   * @brief Compute covariance matrix for a single gene (without caching)
   * @param gene_idx Gene index
   * @return Covariance matrix (n_params × n_params) in row-major order
   */
  std::vector<double> covariance_matrix_single(int gene_idx) const;

 private:
  std::vector<double> compute_se_from_weights(const double* w) const;
  std::vector<double> compute_cov_from_weights(const double* w) const;

  int n_genes_;
  int n_samples_;
  int n_params_;
  const double* X_;  // Design matrix (n_samples × n_params)

  // Basic IRLS results
  std::vector<double> beta_;     // (n_genes × n_params)
  std::vector<double> weights_;  // (n_genes × n_samples) - final IRLS weights
  std::vector<uint8_t> converged_;  // (n_genes)

  // Cached statistics (lazy evaluation with std::optional)
  mutable std::optional<std::vector<double>> se_cache_;  // (n_genes × n_params)
  mutable std::optional<std::vector<double>>
      cov_cache_;  // (n_genes × n_params × n_params)
};

/**
 * @brief Iteratively Reweighted Least Squares solver for GLMs
 */
class IRLSSolver {
 public:
  struct Options {
    int n_threads = 4;
    int max_iter = 100;
    double tol = 1e-8;

    static Options defaults() { return Options{}; }
  };

  IRLSSolver(std::shared_ptr<Distribution> dist,        // exponential family
             const double* X,                           // design matrix
             const double* sf,                          // size factors
             int n_genes, int n_samples, int n_params,  // dimensions
             Options opts = Options::defaults()         // options
             )
      : dist_(std::move(dist)),
        X_(X),
        sf_(sf),
        n_genes_(n_genes),
        n_samples_(n_samples),
        n_params_(n_params),
        opts_(opts) {}

  /**
   * @brief Fit GLM for all genes using IRLS
   * @param Y Count matrix (n_genes × n_samples) in row-major order
   * @return IRLSResult containing beta and final weights
   */
  IRLSResult fit(const double* Y);

 private:
  std::shared_ptr<Distribution> dist_;
  const double* X_;   // Design matrix
  const double* sf_;  // Size factors
  int n_genes_;
  int n_samples_;
  int n_params_;
  Options opts_;

  static void* thread_worker(void* arg);

  /**
   * @brief Fit a single gene using IRLS
   * @param gene_idx Gene index (for distribution variance lookup)
   * @param y Counts for this gene (n_samples)
   * @param beta_out Output buffer for beta (n_params)
   * @param w_out Output buffer for final weights (n_samples)
   * @param workspace Preallocated workspace
   * @return true if converged, false otherwise
   */
  bool fit_single(int gene_idx, const double* y, double* beta_out,
                  double* w_out, double* workspace);
};

}  // namespace death

#endif  // IRLSSOLVER_HPP