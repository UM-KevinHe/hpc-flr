#!/usr/bin/env Rscript
# fit_flr.R — dispatcher for the fit_flr tool: paraflr::logis_firth().
#
# Fits Firth bias-reduced logistic regression with one intercept per provider
# and no global intercept, optionally across `threads` OpenMP threads.
#
# input.json : {"data_path", "y_expr", "z_expr", "id_expr",
#               "threads", "cutoff", "tol", "max_iter", "bound", "backtrack"}
# output.json: {"status":"ok", "beta": {...}, "gamma": {...}, ...}

.script_dir <- function() {
  f <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  # Rscript encodes spaces in --file= as "~+~" (paths with spaces hit this).
  if (length(f)) dirname(gsub("~\\+~", " ", sub("^--file=", "", f[1]))) else getwd()
}
source(file.path(.script_dir(), "_common.R"))
suppressPackageStartupMessages(library(paraflr))

a <- parse_args()
result <- with_error_payload("fit_flr.R", {
  input  <- read_input(a$input)
  e      <- load_data_env(input$data_path)
  fields <- resolve_flr_fields(input, e)
  ctl    <- fit_controls(input)

  t0  <- proc.time()[["elapsed"]]
  fit <- run_fit(fields, ctl)
  elapsed <- proc.time()[["elapsed"]] - t0

  # Provider effects are returned in full only when there are few enough to
  # be readable; beyond that the caller gets the distribution plus the
  # extremes, which is what provider profiling actually looks at. The full
  # vector always stays available by rerunning repro.R.
  m <- length(fit$gamma)
  gamma_out <- if (m <= 200) as.list(fit$gamma) else NULL
  ord <- order(fit$gamma)
  extremes <- list(
    lowest  = as.list(fit$gamma[head(ord, 5)]),
    highest = as.list(fit$gamma[rev(tail(ord, 5))])
  )

  list(status = "ok",
       beta = as.list(fit$beta),
       gamma = gamma_out,
       gamma_summary = list(
         n_providers = m,
         median = unname(stats::median(fit$gamma)),
         quantiles = as.list(round(stats::quantile(
           fit$gamma, c(0, 0.25, 0.5, 0.75, 1)), 6)),
         extremes = extremes),
       neg2Loglkd = fit$neg2Loglkd,
       n_obs = nrow(fit$data),
       n_covariates = ncol(fields$Z),
       n_providers = m,
       event_rate = round(mean(fit$data$Y), 4),
       threads = ctl$threads,
       elapsed_sec = round(elapsed, 3))
})
write_result(result, a$output)
