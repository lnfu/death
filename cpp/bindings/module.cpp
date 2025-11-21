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

#include "IRLSSolver.hpp"
#include "OLSSolver.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
  m.doc() = "Core module for death package";

  py::class_<OLSSolver::Options>(m, "OLSOptions")
      .def(py::init<>())
      .def("__repr__",
           [](const OLSSolver::Options& opts) { return "<OLSOptions>"; });

  py::class_<OLSSolver::Result>(m, "OLSResult")
      .def_property_readonly(
          "coefficients",
          [](OLSSolver::Result& self) -> py::array_t<double> {
            auto* vec = new std::vector<double>(std::move(self.coefficients));
            py::capsule cleanup(vec, [](void* ptr) {
              delete static_cast<std::vector<double>*>(ptr);
            });

            size_t p = vec->size() / self.nrhs;
            return py::array_t<double>({p, (size_t)self.nrhs}, vec->data(),
                                       cleanup);
          })
      .def_property_readonly(
          "fitted_values",
          [](OLSSolver::Result& self) -> py::array_t<double> {
            auto* vec = new std::vector<double>(std::move(self.fitted_values));
            py::capsule cleanup(vec, [](void* ptr) {
              delete static_cast<std::vector<double>*>(ptr);
            });

            size_t n = vec->size() / self.nrhs;
            return py::array_t<double>({n, (size_t)self.nrhs}, vec->data(),
                                       cleanup);
          })
      .def("__repr__", [](const OLSSolver::Result&) { return "<OLSResult>"; });

  py::class_<OLSSolver>(m, "OLSSolver")
      .def(py::init<>(), "Initialize OLS Solver")
      .def(
          "solve",
          [](OLSSolver& self,                            //
             py::array_t<double, py::array::c_style> X,  //
             py::array_t<double, py::array::c_style> Y,  //
             py::object opts_obj) {
            auto X_buf = X.request();
            auto Y_buf = Y.request();

            if (X_buf.ndim != 2)
              throw std::runtime_error("X must be 2-dimensional");
            if (Y_buf.ndim != 2)
              throw std::runtime_error("Y must be 2-dimensional");

            int n = static_cast<int>(X_buf.shape[0]);
            int p = static_cast<int>(X_buf.shape[1]);
            int nrhs = static_cast<int>(Y_buf.shape[1]);

            if (Y_buf.shape[0] != n)
              throw std::runtime_error("X and Y must have same number of rows");

            const double* X_ptr = static_cast<const double*>(X_buf.ptr);
            const double* Y_ptr = static_cast<const double*>(Y_buf.ptr);

            OLSSolver::Options opts;
            if (!opts_obj.is_none()) {
              opts = opts_obj.cast<OLSSolver::Options>();
            }

            return self.solve(X_ptr, Y_ptr, n, p, nrhs, opts);
          },
          py::arg("X").noconvert(),      //
          py::arg("Y").noconvert(),      //
          py::arg("opts") = py::none(),  //
          R"pbdoc(
Solve an OLS problem.

Parameters
----------
X : numpy.ndarray, shape (n, p)
    Design matrix (must be C-contiguous)
Y : numpy.ndarray, shape (n, nrhs)
    Response matrix (must be C-contiguous)
opts : OLSOptions, optional
    Solver options. If None, uses default options.

Returns
-------
OLSResult
    Solving results with coefficients and fitted values
          )pbdoc");
  //   py::class_<IRLSSolver::Options>(m, "IRLSOptions")
  //       .def(py::init<>())
  //       .def_readwrite("max_iter", &IRLSSolver::Options::max_iter)
  //       .def_readwrite("tol", &IRLSSolver::Options::tol);

  //   py::class_<IRLSSolver::Result>(m, "IRLSResult")
  //       .def_readonly("coefficients", &IRLSSolver::Result::coefficients)
  //       .def_readonly("fitted_values", &IRLSSolver::Result::fitted_values)
  //       .def_readonly("std_errors", &IRLSSolver::Result::std_errors)
  //       .def_readonly("deviance", &IRLSSolver::Result::deviance)
  //       .def_readonly("iterations", &IRLSSolver::Result::iterations)
  //       .def_readonly("converged", &IRLSSolver::Result::converged);

  //   py::class_<IRLSSolver>(m, "IRLSSolver")
  //       .def(py::init<>(), "Initialize IRLS Solver")
  //       .def(
  //           "fit",
  //           [](IRLSSolver& self,
  //              py::array_t<double, py::array::c_style | py::array::forcecast>
  //              X, py::array_t<double, py::array::c_style |
  //              py::array::forcecast> y, const IRLSSolver::Options& opts) {
  //             // Get buffer info
  //             py::buffer_info X_buf = X.request();
  //             py::buffer_info y_buf = y.request();

  //             // Validate dimensions
  //             if (X_buf.ndim != 2)
  //               throw std::runtime_error("X must be 2-dimensional (n x p)");
  //             if (y_buf.ndim != 1)
  //               throw std::runtime_error("y must be 1-dimensional");

  //             // Extract sizes
  //             int n = static_cast<int>(X_buf.shape[0]);
  //             int p = static_cast<int>(X_buf.shape[1]);

  //             // Validate size match
  //             if (y_buf.shape[0] != n)
  //               throw std::runtime_error("X and y must have same number of
  //               rows");

  //             // Zero-copy: directly use pointers
  //             const double* X_ptr = static_cast<const double*>(X_buf.ptr);
  //             const double* y_ptr = static_cast<const double*>(y_buf.ptr);

  //             return self.solve(X_ptr, n, p, y_ptr, opts);
  //           },
  //           py::arg("X").noconvert(),  // Avoid unnecessary conversion
  //           py::arg("y").noconvert(),  // Avoid unnecessary conversion
  //           py::arg_v("opts", IRLSSolver::Options(), "IRLSOptions()"),
  //           R"pbdoc(
  // Fit an IRLS model.

  // Parameters
  // ----------
  // X : numpy.ndarray, shape (n, p)
  //     Design matrix
  // y : numpy.ndarray, shape (n,)
  //     Response vector
  // opts : IRLSOptions, optional
  //     Solver options

  // Returns
  // -------
  // IRLSResult
  //     Fitting results
  // )pbdoc");

  //   m.attr("__version__") = "0.1.0";
}
