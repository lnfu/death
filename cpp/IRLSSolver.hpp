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
#include <vector>

#include "Distribution.hpp"

class IRLSSolver {
 private:
  std::unique_ptr<Distribution> dist_;

 public:
  struct Options {
    int max_iter = 100;
    double tol = 1e-8;

    // bool use_offset = false;
    // bool estimate_dispersion = false;

    static Options defaults() { return Options{}; }
  };

  struct Result {
    std::vector<double> coefficients;
    std::vector<double> fitted_values;
    std::vector<double> std_errors;
    double deviance;
    int iterations;
    bool converged;
  };

  // TODO(Enfu) remove this default constructor
  IRLSSolver() : dist_(std::make_unique<NormalDistribution>()) {}

  IRLSSolver(std::unique_ptr<Distribution> dist) : dist_(std::move(dist)) {}

  Result fit(const double* X, int n, int p,  // design matrix
             const double* y,                // response
             // const double *offset = nullptr, // optional offset
             const Options& opts = Options::defaults());
};

#endif  // IRLSSOLVER_HPP
