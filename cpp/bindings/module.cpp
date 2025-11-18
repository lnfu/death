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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "Distribution.hpp"
#include "IRLSSolver.hpp"

namespace py = pybind11;

namespace death {

// Helper function to validate numpy arrays
template <typename T>
void validate_array(py::array_t<T> arr, const std::string& name,
                    const std::vector<ssize_t>& expected_shape = {}) {
  if (!expected_shape.empty()) {
    auto shape = arr.shape();
    if (arr.ndim() != expected_shape.size()) {
      throw std::runtime_error(name + " has wrong number of dimensions");
    }
    for (size_t i = 0; i < expected_shape.size(); ++i) {
      if (expected_shape[i] != -1 && shape[i] != expected_shape[i]) {
        throw std::runtime_error(name + " has wrong shape at dimension " +
                                 std::to_string(i));
      }
    }
  }

  if (!(arr.flags() & py::array::c_style)) {
    throw std::runtime_error(name + " must be C-contiguous");
  }
}

// Helper to create numpy array from std::vector with zero-copy
template <typename T>
py::array_t<T> vector_to_numpy(std::vector<T>&& vec,
                               const std::vector<ssize_t>& shape) {
  // Transfer ownership to a new heap-allocated vector
  std::vector<T>* vec_ptr = new std::vector<T>(std::move(vec));

  // Create capsule for cleanup
  auto capsule = py::capsule(
      vec_ptr, [](void* ptr) { delete static_cast<std::vector<T>*>(ptr); });

  // Calculate strides (row-major)
  std::vector<ssize_t> strides(shape.size());
  ssize_t stride = sizeof(T);
  for (int i = shape.size() - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }

  // Create numpy array with capsule
  return py::array_t<T>(shape, strides, vec_ptr->data(), capsule);
}

}  // namespace death

PYBIND11_MODULE(_core, m) {
  m.doc() =
      "DEATH: Differential Expression Analysis Tool for High-throughput NGS";

  // ==================== Distribution Classes ====================

  py::class_<death::Distribution, std::shared_ptr<death::Distribution>>(
      m, "Distribution")
      .def("link", &death::Distribution::link, py::arg("mu"),
           "Link function g(μ)")
      .def("inverse_link", &death::Distribution::inverse_link, py::arg("eta"),
           "Inverse link function g⁻¹(η)")
      .def("derivative_link", &death::Distribution::derivative_link,
           py::arg("mu"), "Derivative of link function g'(μ)")
      .def("variance", &death::Distribution::variance, py::arg("mu"),
           py::arg("idx"), "Variance function V(μ)");

  py::class_<death::NormalDistribution, death::Distribution,
             std::shared_ptr<death::NormalDistribution>>(m,
                                                         "NormalDistribution")
      .def(py::init<>(), "Normal distribution with identity link");

  py::class_<death::PoissonDistribution, death::Distribution,
             std::shared_ptr<death::PoissonDistribution>>(m,
                                                          "PoissonDistribution")
      .def(py::init<>(), "Poisson distribution with log link");

  py::class_<death::BernoulliDistribution, death::Distribution,
             std::shared_ptr<death::BernoulliDistribution>>(
      m, "BernoulliDistribution")
      .def(py::init<>(), "Bernoulli distribution with logit link");

  py::class_<death::NegativeBinomialDistribution, death::Distribution,
             std::shared_ptr<death::NegativeBinomialDistribution>>(
      m, "NegativeBinomialDistribution")
      .def(py::init([](py::array_t<double> dispersions) {
             auto buf = dispersions.request();
             if (buf.ndim != 1) {
               throw std::runtime_error("dispersions must be 1-dimensional");
             }
             const double* disp_ptr = static_cast<const double*>(buf.ptr);
             return std::make_shared<death::NegativeBinomialDistribution>(
                 disp_ptr);
           }),
           py::arg("dispersions"), py::keep_alive<1, 2>(),
           "Negative binomial distribution with log link");

  // ==================== IRLSResult Class ====================

  py::class_<death::IRLSResult>(
      m, "IRLSResult",
      "Results from IRLS fitting with lazy-evaluated statistics.\n\n"
      "This class stores the basic fitting results (beta, weights) and\n"
      "provides on-demand computation of statistical quantities (SE, Cov).\n"
      "Following the design pattern of statsmodels.GLMResults.")
      // Basic properties
      .def_property_readonly("n_genes", &death::IRLSResult::n_genes,
                             "Number of genes")
      .def_property_readonly("n_samples", &death::IRLSResult::n_samples,
                             "Number of samples")
      .def_property_readonly("n_params", &death::IRLSResult::n_params,
                             "Number of parameters")

      // Basic results (always available, no computation needed)
      .def_property_readonly(
          "beta",
          [](death::IRLSResult& self) -> py::array_t<double> {
            const auto& beta = self.beta();
            return py::array_t<double>(
                {self.n_genes(), self.n_params()},                   // shape
                {self.n_params() * sizeof(double), sizeof(double)},  // strides
                beta.data(),    // data pointer
                py::cast(self)  // parent object (keep alive)
            );
          },
          "Fitted coefficients (n_genes × n_params), zero-copy view")

      .def_property_readonly(
          "weights",
          [](death::IRLSResult& self) -> py::array_t<double> {
            const auto& weights = self.weights();
            return py::array_t<double>(
                {self.n_genes(), self.n_samples()},
                {self.n_samples() * sizeof(double), sizeof(double)},
                weights.data(), py::cast(self));
          },
          "Final IRLS weights (n_genes × n_samples), zero-copy view")

      .def_property_readonly(
          "converged",
          [](death::IRLSResult& self) -> py::array_t<uint8_t> {
            const auto& converged = self.converged();
            return py::array_t<uint8_t>({self.n_genes()}, {sizeof(uint8_t)},
                                        converged.data(), py::cast(self));
          },
          "Convergence flags (n_genes), zero-copy view")

      // Lazy-evaluated statistics (computed on first access, then cached)
      .def_property_readonly(
          "standard_errors",
          [](death::IRLSResult& self) -> py::array_t<double> {
            const auto& se = self.standard_errors();
            return py::array_t<double>(
                {self.n_genes(), self.n_params()},
                {self.n_params() * sizeof(double), sizeof(double)}, se.data(),
                py::cast(self));
          },
          "Standard errors (n_genes × n_params).\n"
          "Computed on first access and cached.\n"
          "Zero-copy view of cached result.")

      .def_property_readonly(
          "covariance_matrices",
          [](death::IRLSResult& self) -> py::array_t<double> {
            const auto& cov = self.covariance_matrices();
            return py::array_t<double>(
                {self.n_genes(), self.n_params(), self.n_params()},
                {self.n_params() * self.n_params() * sizeof(double),
                 self.n_params() * sizeof(double), sizeof(double)},
                cov.data(), py::cast(self));
          },
          "Covariance matrices (n_genes × n_params × n_params).\n"
          "Computed on first access and cached.\n"
          "Zero-copy view of cached result.")

      // Single gene methods (no caching, useful for specific genes)
      .def(
          "standard_errors_single",
          [](death::IRLSResult& self, int gene_idx) -> py::array_t<double> {
            auto se = self.standard_errors_single(gene_idx);
            return death::vector_to_numpy(std::move(se), {self.n_params()});
          },
          py::arg("gene_idx"),
          "Compute SE for a single gene (no caching).\n\n"
          "Parameters:\n"
          "  gene_idx: Gene index\n\n"
          "Returns:\n"
          "  numpy array of shape (n_params,)")

      .def(
          "covariance_matrix_single",
          [](death::IRLSResult& self, int gene_idx) -> py::array_t<double> {
            auto cov = self.covariance_matrix_single(gene_idx);
            return death::vector_to_numpy(std::move(cov),
                                          {self.n_params(), self.n_params()});
          },
          py::arg("gene_idx"),
          "Compute covariance matrix for a single gene (no caching).\n\n"
          "Parameters:\n"
          "  gene_idx: Gene index\n\n"
          "Returns:\n"
          "  numpy array of shape (n_params, n_params)")

      // Utility method
      .def("__repr__", [](const death::IRLSResult& self) -> std::string {
        int n_converged =
            std::count(self.converged().begin(), self.converged().end(), 1);
        return "<IRLSResult: " + std::to_string(self.n_genes()) + " genes, " +
               std::to_string(self.n_params()) + " params, " +
               std::to_string(n_converged) + " converged>";
      });

  // ==================== IRLSSolver::Options ====================

  py::class_<death::IRLSSolver::Options>(m, "IRLSSolverOptions")
      .def(py::init<>())
      .def(py::init([](py::dict d) {
             auto opts = death::IRLSSolver::Options::defaults();
             if (d.contains("n_threads"))
               opts.n_threads = d["n_threads"].cast<int>();
             if (d.contains("max_iter"))
               opts.max_iter = d["max_iter"].cast<int>();
             if (d.contains("tol")) opts.tol = d["tol"].cast<double>();
             return opts;
           }),
           py::arg("dict"), "Create options from dictionary")
      .def_readwrite("n_threads", &death::IRLSSolver::Options::n_threads,
                     "Number of threads to use")
      .def_readwrite("max_iter", &death::IRLSSolver::Options::max_iter,
                     "Maximum number of iterations")
      .def_readwrite("tol", &death::IRLSSolver::Options::tol,
                     "Convergence tolerance")
      .def_static("defaults", &death::IRLSSolver::Options::defaults,
                  "Get default options")
      .def("__repr__",
           [](const death::IRLSSolver::Options& opts) -> std::string {
             return "<IRLSSolverOptions: n_threads=" +
                    std::to_string(opts.n_threads) +
                    ", max_iter=" + std::to_string(opts.max_iter) +
                    ", tol=" + std::to_string(opts.tol) + ">";
           });

  // ==================== IRLSSolver Class ====================

  py::class_<death::IRLSSolver>(
      m, "IRLSSolver",
      "Iteratively Reweighted Least Squares solver for GLMs.\n\n"
      "This class performs the numerical optimization (IRLS iteration).\n"
      "Statistical inference (SE, p-values) is handled by IRLSResult.")
      .def(py::init([](std::shared_ptr<death::Distribution> dist,
                       py::array_t<double> X, py::array_t<double> sf,
                       int n_genes, int n_samples, int n_params,
                       py::object opts_obj) {
             // Validate inputs
             death::validate_array(X, "X", {n_samples, n_params});
             death::validate_array(sf, "sf", {n_samples});

             auto X_buf = X.request();
             auto sf_buf = sf.request();

             const double* X_ptr = static_cast<const double*>(X_buf.ptr);
             const double* sf_ptr = static_cast<const double*>(sf_buf.ptr);

             // Handle optional opts parameter
             death::IRLSSolver::Options opts;
             if (!opts_obj.is_none()) {
               opts = opts_obj.cast<death::IRLSSolver::Options>();
             } else {
               opts = death::IRLSSolver::Options::defaults();
             }

             return std::make_unique<death::IRLSSolver>(
                 dist, X_ptr, sf_ptr, n_genes, n_samples, n_params, opts);
           }),
           py::arg("dist"), py::arg("X"), py::arg("sf"), py::arg("n_genes"),
           py::arg("n_samples"), py::arg("n_params"),
           py::arg("opts") = py::none(),
           py::keep_alive<1, 2>(),  // Keep dist alive
           py::keep_alive<1, 3>(),  // Keep X alive
           py::keep_alive<1, 4>(),  // Keep sf alive
           "Initialize IRLS solver\n\n"
           "Parameters:\n"
           "  dist: Distribution object\n"
           "  X: Design matrix (n_samples × n_params)\n"
           "  sf: Size factors (n_samples)\n"
           "  n_genes: Number of genes\n"
           "  n_samples: Number of samples\n"
           "  n_params: Number of parameters\n"
           "  opts: Solver options (optional, defaults to defaults())\n\n"
           "Example:\n"
           "  >>> dist = NegativeBinomialDistribution(dispersions)\n"
           "  >>> solver = IRLSSolver(dist, X, sf, 1000, 6, 2)\n"
           "  >>> result = solver.fit(counts)")

      .def(
          "fit",
          [](death::IRLSSolver& self,
             py::array_t<double> Y_array) -> death::IRLSResult {
            // Validate input
            death::validate_array(Y_array, "Y");

            if (Y_array.ndim() != 2) {
              throw std::runtime_error(
                  "Y must be a 2D array (n_genes × n_samples)");
            }

            // Get pointer to data
            auto Y_buf = Y_array.request();
            const double* Y = static_cast<const double*>(Y_buf.ptr);

            // Call C++ fit method (returns IRLSResult by value)
            return self.fit(Y);
          },
          py::arg("Y"),
          "Fit the GLM model for all genes.\n\n"
          "Parameters:\n"
          "  Y: Count matrix (n_genes × n_samples)\n\n"
          "Returns:\n"
          "  IRLSResult object with beta, weights, and lazy-evaluated "
          "statistics\n\n"
          "Example:\n"
          "  >>> result = solver.fit(counts)\n"
          "  >>> print(result.beta)  # Available immediately\n"
          "  >>> print(result.standard_errors)  # Computed on first access");

  // Version info
  m.attr("__version__") = "0.1.0";
}