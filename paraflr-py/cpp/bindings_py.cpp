// bindings_py.cpp — pybind11 wrapper around the flr:: core. Matrix inputs are
// forced to Fortran (column-major) order to match Armadillo.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <armadillo>
#include <cstring>
#include <string>
#include "firth_core.hpp"

namespace py = pybind11;

using ArrC = py::array_t<double, py::array::c_style | py::array::forcecast>;
using ArrF = py::array_t<double, py::array::f_style | py::array::forcecast>;

static arma::vec to_vec(ArrC a) {
  auto buf = a.request();
  if (buf.ndim != 1)
    throw std::runtime_error("expected a 1-D array");
  return arma::vec(static_cast<double*>(buf.ptr),
                   static_cast<arma::uword>(buf.shape[0]));  // copies
}

static arma::mat to_mat(ArrF a) {
  auto buf = a.request();
  if (buf.ndim != 2)
    throw std::runtime_error("expected a 2-D array");
  return arma::mat(static_cast<double*>(buf.ptr),
                   static_cast<arma::uword>(buf.shape[0]),
                   static_cast<arma::uword>(buf.shape[1]));  // f_style -> column-major, copies
}

static py::array_t<double> to_np(const arma::vec& v) {
  py::array_t<double> out(static_cast<py::ssize_t>(v.n_elem));
  std::memcpy(out.mutable_data(), v.memptr(), v.n_elem * sizeof(double));
  return out;
}

PYBIND11_MODULE(_core, m) {
  m.doc() = "paraflr C++ core (shared with the R package) via pybind11";

  m.def("loglkd", [](ArrC Y, ArrC Z_beta, ArrC gamma_obs) {
    return flr::Loglkd(to_vec(Y), to_vec(Z_beta), to_vec(gamma_obs));
  });

  m.def("logis_firth_prov",
    [](ArrC Y, ArrF Z, ArrC n_prov, ArrC gamma, ArrC beta,
       int n_obs, int m, int threads, double tol, int max_iter,
       double bound, bool message, const std::string& stop, bool need_trace) {
      arma::vec Yv = to_vec(Y), npv = to_vec(n_prov),
                gv = to_vec(gamma), bv = to_vec(beta);
      arma::mat Zv = to_mat(Z);
      flr::FirthFit f = flr::logis_firth_prov(
          Yv, Zv, npv, gv, bv, n_obs, m, threads, tol, max_iter, bound,
          message, stop, need_trace);
      py::dict d;
      d["gamma"]  = to_np(f.gamma);
      d["beta"]   = to_np(f.beta);
      d["loglik"] = f.loglik;
      d["iter"]   = f.iter;
      d["crit"]   = f.crit;
      return d;
    },
    py::arg("Y"), py::arg("Z"), py::arg("n_prov"), py::arg("gamma"),
    py::arg("beta"), py::arg("n_obs"), py::arg("m"), py::arg("threads") = 1,
    py::arg("tol") = 1e-5, py::arg("max_iter") = 100, py::arg("bound") = 10.0,
    py::arg("message") = false, py::arg("stop") = "beta",
    py::arg("need_trace") = false);

  m.def("loglkd_firth", [](ArrC Y, ArrF Z, ArrC n_prov, ArrC gamma, ArrC beta) {
    arma::vec Yv = to_vec(Y), npv = to_vec(n_prov),
              gv = to_vec(gamma), bv = to_vec(beta);
    arma::mat Zv = to_mat(Z);
    return flr::Loglkd_firth(Yv, Zv, npv, gv, bv);
  });

  m.def("wald_gamma", [](ArrF Z, ArrC p, ArrC n_prov, ArrC parm_indices) {
    arma::mat Zv = to_mat(Z);
    arma::vec pv = to_vec(p), npv = to_vec(n_prov), piv = to_vec(parm_indices);
    return to_np(flr::wald_gamma(Zv, pv, npv, piv));
  });
}
