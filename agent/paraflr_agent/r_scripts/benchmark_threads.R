#!/usr/bin/env Rscript
# benchmark_threads.R — dispatcher for the benchmark_threads tool.
#
# Refits the same model at several OpenMP thread counts and reports wall time,
# speedup, and the largest coefficient discrepancy across the runs. The last
# of those is the point: shared-memory parallelism must not move the estimate,
# so a non-negligible max_beta_diff is a bug report, not a timing curiosity.
#
# input.json : {"data_path", "y_expr", "z_expr", "id_expr",
#               "threads_list": [1, 2, 4, 8], "repeats", plus fit controls}
# output.json: {"status":"ok", "openmp": bool, "runs": [...], ...}

.script_dir <- function() {
  f <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  # Rscript encodes spaces in --file= as "~+~" (paths with spaces hit this).
  if (length(f)) dirname(gsub("~\\+~", " ", sub("^--file=", "", f[1]))) else getwd()
}
source(file.path(.script_dir(), "_common.R"))
suppressPackageStartupMessages(library(paraflr))

a <- parse_args()
result <- with_error_payload("benchmark_threads.R", {
  input  <- read_input(a$input)
  e      <- load_data_env(input$data_path)
  fields <- resolve_flr_fields(input, e)
  ctl    <- fit_controls(input)

  threads_list <- if (is.null(input$threads_list)) c(1L, 2L, 4L) else
    as.integer(unlist(input$threads_list))
  threads_list <- unique(threads_list[threads_list >= 1L])
  if (!length(threads_list)) stop("threads_list must contain at least one thread count >= 1")
  repeats <- if (is.null(input$repeats)) 1L else max(1L, as.integer(input$repeats))

  runs <- list()
  beta_ref <- NULL
  max_diff <- 0
  for (th in threads_list) {
    times <- numeric(repeats)
    for (r in seq_len(repeats)) {
      ctl_th <- ctl
      ctl_th$threads <- as.integer(th)
      t0 <- proc.time()[["elapsed"]]
      fit <- run_fit(fields, ctl_th)
      times[r] <- proc.time()[["elapsed"]] - t0
    }
    if (is.null(beta_ref)) {
      beta_ref <- fit$beta
    } else {
      max_diff <- max(max_diff, max(abs(fit$beta - beta_ref)))
    }
    runs[[length(runs) + 1L]] <- list(
      threads = as.integer(th),
      elapsed_sec = round(min(times), 4),
      elapsed_sec_all = as.list(round(times, 4)))
  }

  base <- runs[[1]]$elapsed_sec
  for (i in seq_along(runs)) {
    runs[[i]]$speedup <- round(base / runs[[i]]$elapsed_sec, 3)
    runs[[i]]$efficiency <- round(
      (base / runs[[i]]$elapsed_sec) / runs[[i]]$threads, 3)
  }

  fastest <- runs[[which.min(vapply(runs, function(r) r$elapsed_sec, numeric(1)))]]

  # A build without OpenMP runs the same serial code at every thread count,
  # so flat timings are a build report, not a scaling result. We do not try to
  # detect the build flag; we report the observation and let the caller say so.
  list(status = "ok",
       parallel_speedup_observed = fastest$speedup > 1.2,
       detected_cores = parallel::detectCores(),
       threads_list = as.list(threads_list),
       repeats = repeats,
       runs = runs,
       baseline_threads = runs[[1]]$threads,
       fastest_threads = fastest$threads,
       best_speedup = fastest$speedup,
       max_beta_diff = signif(max_diff, 6),
       estimates_agree = max_diff < 1e-8,
       n_obs = length(fields$Y),
       n_providers = length(unique(fields$ID)),
       n_covariates = ncol(fields$Z))
})
write_result(result, a$output)
