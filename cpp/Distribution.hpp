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

#ifndef DISTRIBUTION_HPP
#define DISTRIBUTION_HPP

#include <cmath>

class Distribution {
 private:
  /* data */
 public:
  virtual ~Distribution() = default;

  virtual double link(double mu) const = 0;             // g(μ)
  virtual double inverse_link(double eta) const = 0;    // g^(-1)(η)
  virtual double derivative_link(double mu) const = 0;  // g'(μ)

  virtual double variance(double mu) const = 0;  // V(μ)
};

class NormalDistribution : public Distribution {
  // Identity link
  double link(double mu) const override { return mu; }

  double inverse_link(double eta) const override { return eta; }

  double derivative_link(double mu) const override { return 1.0; }

  // Constant variance
  double variance(double mu) const override { return 1.0; }
};

class PoissonDistribution : public Distribution {
  // Log link
  double link(double mu) const override { return std::log(mu); }

  double inverse_link(double eta) const override { return std::exp(eta); }
  double derivative_link(double mu) const override { return 1.0 / mu; }
  // Variance = μ
  double variance(double mu) const override { return mu; }
};

class BernoulliDistribution : public Distribution {
  // Logit link
  double link(double mu) const override { return std::log(mu / (1.0 - mu)); }
  double inverse_link(double eta) const override {
    return 1.0 / (1.0 + std::exp(-eta));
  }
  double derivative_link(double mu) const override {
    return 1.0 / (mu * (1.0 - mu));
  }
  // Variance = μ(1-μ)
  double variance(double mu) const override { return mu * (1.0 - mu); }
};

class NegativeBinomialDistribution : public Distribution {
 private:
  double dispersion_;

 public:
  // TODO(Enfu): Another link function is possible for Negative Binomial
  //             so we may want to make it configurable (or just move to
  //             another class)

  double link(double mu) const override { return std::log(mu); }

  double inverse_link(double eta) const override { return std::exp(eta); }

  double derivative_link(double mu) const override { return 1.0 / mu; }

  double variance(double mu) const override {
    return mu + (mu * mu) / dispersion_;
  }
};

#endif  // DISTRIBUTION_HPP
