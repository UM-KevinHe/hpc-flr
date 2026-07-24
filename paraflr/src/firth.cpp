#define STRICT_R_HEADERS
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <cmath>
#include <cfloat>
#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::plugins(cpp14)]]
// [[Rcpp::plugins(openmp)]]
using namespace Rcpp;
using namespace arma;
using namespace std;

arma::vec rep(arma::vec &x, arma::vec &each) {
  arma::vec x_rep(sum(each));
  int ind = 0, m = x.n_elem;
  for (int i = 0; i < m; i++) {
    x_rep.subvec(ind, ind + each(i) - 1) = x(i) * ones(each(i));
    ind += each(i);
  }
  return x_rep;
}

// [[Rcpp::export]]
double Loglkd(const arma::vec &Y, const arma::vec &Z_beta, const arma::vec &gamma_obs) {
  arma::vec eta = gamma_obs + Z_beta;
  arma::vec softplus = arma::clamp(eta, 0.0, arma::datum::inf) +
                       arma::log(1.0 + arma::exp(-arma::abs(eta)));
  return arma::sum(eta % Y - softplus);
}

void ind2uppsub(unsigned int index, unsigned int dim, unsigned int &row, unsigned int &col) {
  row = 0;
  col = dim - 1;
  unsigned int n = dim * (dim - 1) / 2 - (dim - row) * (dim - row - 1) / 2 + col;
  while (index > n) {
    ++row;
    n = dim * (dim - 1) / 2 - (dim - row) * (dim - row - 1) / 2 + col;
  }
  while (index < n) {
    --col;
    --n;
  }
}

arma::mat info_beta_omp(const arma::mat &Z, const arma::vec &pq, const int &threads) {
  unsigned int p = Z.n_cols;
  unsigned int loops = p * (1 + p) / 2;
  arma::mat output(p, p);
  #pragma omp parallel for schedule(static)
  for (unsigned int i = 0; i < loops; i++) {
    unsigned int r, c;
    ind2uppsub(i, p, r, c);
    output(r, c) = dot(Z.col(r), Z.col(c) % pq);
    output(c, r) = output(r, c);
  }
  return output;
}

// [[Rcpp::export]]
List logis_firth_prov(arma::vec &Y, arma::mat &Z, arma::vec &n_prov, arma::vec gamma, arma::vec beta,
                      int n_obs, int m, const int &threads, double tol = 1e-5, int max_iter = 10000,
                      double bound = 10.0, bool message = true, bool backtrack = false) {
  int iter = 0, n = n_obs, ind = 0;
  double v, crit = 100.0;
  double s = 0.01, t = 0.6, lambda = 0.0, d_loglkd = 0.0, loglkd = 0.0;
  arma::vec gamma_obs(n), gamma_obs_tmp(n), gamma_tmp(m);

  arma::ivec indices(m + 1);
  for (int i = 0; i < m; i++) {
    indices(i) = ind;
    ind += n_prov(i);
  }
  indices(m) = ind;

  while (iter < max_iter) {
    if (crit < tol) break;
    iter++;
    gamma_obs = rep(gamma, n_prov);
    arma::vec Z_beta = Z * beta;
    arma::vec p = 1 / (1 + exp(-gamma_obs - Z_beta));
    arma::vec Yp = Y - p, pq = p % (1 - p);
    if (any(pq == 0)) pq.replace(0, 1e-20);

    arma::vec score_gamma(m), info_gamma_inv(m);
    arma::mat info_betagamma(Z.n_cols, m);
    for (int i = 0; i < m; i++) {
      info_gamma_inv(i) = 1 / sum(pq.subvec(indices(i), indices(i + 1) - 1));
      info_betagamma.col(i) =
        sum(Z.rows(indices(i), indices(i + 1) - 1).each_col() % pq.subvec(indices(i), indices(i + 1) - 1)).t();
    }

    arma::mat info_beta(Z.n_cols, Z.n_cols);
    if (threads > 1) {
#ifdef _OPENMP
      omp_set_num_threads(threads);
#endif
      info_beta = info_beta_omp(Z, pq, threads);
    } else {
      info_beta = Z.t() * (Z.each_col() % pq);
    }

    arma::mat mat_tmp1 = trans(info_betagamma.each_row() % info_gamma_inv.t());
    arma::mat schur_inv = inv_sympd(info_beta - mat_tmp1.t() * info_betagamma.t());
    arma::mat mat_tmp2 = mat_tmp1 * schur_inv;
    arma::mat prod = mat_tmp1 * schur_inv * mat_tmp1.t();
    arma::vec diag_prod = info_gamma_inv + prod.diag();
    arma::vec c1 = rep(diag_prod, n_prov);

    arma::vec c2(n), c3(n);
    if (threads > 1) {
      #pragma omp parallel for schedule(static)
      for (unsigned int i = 0; i < (unsigned int)m; i++) {
        c2.subvec(indices(i), indices(i + 1) - 1) =
          -Z.rows(indices(i), indices(i + 1) - 1) * mat_tmp2.t().eval().col(i);
      }
      #pragma omp parallel for schedule(static)
      for (unsigned int i = 0; i < (unsigned int)n; i++) {
        c3(i) = (Z.row(i) * schur_inv * Z.row(i).t()).eval()(0, 0);
      }
    } else {
      for (unsigned int i = 0; i < (unsigned int)m; i++) {
        c2.subvec(indices(i), indices(i + 1) - 1) =
          -Z.rows(indices(i), indices(i + 1) - 1) * mat_tmp2.t().eval().col(i);
      }
      for (unsigned int i = 0; i < (unsigned int)n; i++) {
        c3(i) = (Z.row(i) * schur_inv * Z.row(i).t()).eval()(0, 0);
      }
    }

    arma::vec YpA = Yp + pq % (c1 + c2 + c2 + c3) % (0.5 - p);
    arma::vec score_beta = Z.t() * YpA;
    for (int i = 0; i < m; i++) {
      score_gamma(i) = sum(YpA.subvec(indices(i), indices(i + 1) - 1));
    }

    arma::vec d_gamma = info_gamma_inv % score_gamma + mat_tmp2 * (mat_tmp1.t() * score_gamma - score_beta);
    arma::vec d_beta = schur_inv * score_beta - mat_tmp2.t() * score_gamma;

    v = 1.0;
    if (backtrack) {
      loglkd = Loglkd(Y, Z_beta, gamma_obs);
      gamma_tmp = gamma + v * d_gamma;
      gamma_obs_tmp = rep(gamma_tmp, n_prov);
      arma::vec Z_beta_tmp = Z * (beta + v * d_beta);
      d_loglkd = Loglkd(Y, Z_beta_tmp, gamma_obs_tmp) - loglkd;
      lambda = dot(score_gamma, d_gamma) + dot(score_beta, d_beta);
      while (d_loglkd < s * v * lambda) {
        v = t * v;
        gamma_tmp = gamma + v * d_gamma;
        gamma_obs_tmp = rep(gamma_tmp, n_prov);
        Z_beta_tmp = Z * (beta + v * d_beta);
        d_loglkd = Loglkd(Y, Z_beta_tmp, gamma_obs_tmp) - loglkd;
      }
    }

    gamma += v * d_gamma;
    gamma = clamp(gamma, median(gamma) - bound, median(gamma) + bound);
    beta += v * d_beta;
    crit = norm(v * d_beta, "inf");

    if (message) Rcout << "Iter " << iter << ": crit = " << crit << "\n";
  }
  if (message) Rcout << "Firth converged after " << iter << " iterations.\n";
  return List::create(_["gamma"] = gamma, _["beta"] = beta);
}

double logdet_info(const arma::vec &info_gamma, const arma::mat &schur) {
  const double eps = 1e-12;
  const double logdetA = arma::sum(arma::log(arma::clamp(info_gamma, eps, DBL_MAX)));
  arma::mat S = 0.5 * (schur + schur.t());
  arma::mat R;
  if (!arma::chol(R, S)) {
    const double ridge = 1e-8;
    if (!arma::chol(R, S + ridge * arma::eye(S.n_rows, S.n_cols)))
      Rcpp::stop("Cholesky failed: Schur complement not PD.");
  }
  return logdetA + 2.0 * arma::sum(arma::log(R.diag()));
}

// [[Rcpp::export]]
double Loglkd_firth(arma::vec &Y, arma::mat &Z, arma::vec &n_prov, arma::vec &gamma, arma::vec &beta) {
  int m = gamma.n_elem, beta_size = beta.n_elem;
  arma::vec info_gamma(m);
  arma::mat info_betagamma(beta_size, m, arma::fill::zeros);
  arma::mat schur(beta_size, beta_size, arma::fill::zeros);

  arma::vec Z_beta = Z * beta;
  arma::vec gamma_obs = rep(gamma, n_prov);
  arma::vec p = 1 / (1 + exp(-gamma_obs - Z_beta));
  arma::vec pq = p % (1 - p);
  if (any(pq == 0)) pq.replace(0, 1e-10);

  int ind = 0;
  for (int i = 0; i < m; ++i) {
    info_gamma(i) = sum(pq.subvec(ind, ind + n_prov(i) - 1));
    info_betagamma.col(i) =
      sum(Z.rows(ind, ind + n_prov(i) - 1).each_col() % pq.subvec(ind, ind + n_prov(i) - 1)).t();
    ind += n_prov(i);
  }
  arma::mat info_beta = Z.t() * (Z.each_col() % pq);
  arma::vec ig_inv = 1.0 / info_gamma;
  schur = info_beta - (info_betagamma.each_row() % ig_inv.t()) * info_betagamma.t();
  return Loglkd(Y, Z_beta, gamma_obs) + 0.5 * logdet_info(info_gamma, schur);
}

// [[Rcpp::export]]
arma::vec wald_gamma(arma::mat &Z, arma::vec &p, arma::vec &n_prov, arma::vec &parm_indices) {
  int m = n_prov.n_elem, ind = 0, n_p = parm_indices.n_elem;
  arma::vec na_vec(n_p);
  na_vec.fill(arma::datum::nan);

  arma::vec pq = p % (1 - p);
  double eps = 1e-20;
  pq = clamp(pq, eps, arma::datum::inf);

  arma::vec info_gamma_inv(m);
  arma::mat info_betagamma(Z.n_cols, m), info_beta(Z.n_cols, Z.n_cols);

  for (int i = 0; i < m; i++) {
    double s = sum(pq(arma::span(ind, ind + n_prov(i) - 1)));
    if (s < eps) return na_vec;
    info_gamma_inv(i) = 1.0 / s;
    if (!std::isfinite(info_gamma_inv(i))) return na_vec;
    info_betagamma.col(i) =
      sum(Z.rows(ind, ind + n_prov(i) - 1).each_col() % pq.subvec(ind, ind + n_prov(i) - 1)).t();
    ind += n_prov(i);
  }
  info_beta = Z.t() * (Z.each_col() % pq);

  arma::mat J1 = info_betagamma.each_row() % info_gamma_inv.t();
  if (!J1.is_finite()) return na_vec;
  arma::mat schur = info_beta - J1 * info_betagamma.t();
  if (!schur.is_finite()) return na_vec;

  arma::uvec ci = arma::conv_to<arma::uvec>::from(parm_indices) - 1;
  arma::mat sJ1 = J1.cols(ci);
  arma::vec sig = info_gamma_inv(ci);

  arma::vec se_gamma(n_p);
  for (int i = 0; i < n_p; i++) {
    arma::vec J2;
    bool ok = arma::solve(J2, schur, sJ1.col(i), arma::solve_opts::likely_sympd);
    if (!ok) return na_vec;
    double var_i = sig(i) + dot(sJ1.col(i), J2);
    if (!std::isfinite(var_i) || var_i < 0) return na_vec;
    se_gamma(i) = std::sqrt(var_i);
  }
  if (!se_gamma.is_finite()) return na_vec;
  return se_gamma;
}
