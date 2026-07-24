#!/usr/bin/env Rscript
# simulate_provider_data.R — dispatcher for the simulate_provider_data tool.
#
# Generates clustered binary data with provider-specific intercepts and writes
# it to an .rda the other tools can read. This exists because the Medicare
# claims data behind the paper cannot be shared: without it there is no way to
# demonstrate or evaluate the agent end to end.
#
# input.json : {"n_providers", "n_per_provider", "n_covariates", "event_rate",
#               "gamma_sd", "seed", "out_path"}
# output.json: {"status":"ok", "data_path", "object_name", "y_expr", ...}

.script_dir <- function() {
  f <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  # Rscript encodes spaces in --file= as "~+~" (paths with spaces hit this).
  if (length(f)) dirname(gsub("~\\+~", " ", sub("^--file=", "", f[1]))) else getwd()
}
source(file.path(.script_dir(), "_common.R"))

a <- parse_args()
result <- with_error_payload("simulate_provider_data.R", {
  input <- read_input(a$input)
  num <- function(x, d) if (is.null(x)) d else as.numeric(x)
  int <- function(x, d) if (is.null(x)) d else as.integer(x)

  m          <- int(input$n_providers, 50L)
  n_per      <- int(input$n_per_provider, 80L)
  p          <- int(input$n_covariates, 5L)
  event_rate <- num(input$event_rate, 0.2)
  gamma_sd   <- num(input$gamma_sd, 0.5)
  seed       <- int(input$seed, 2026L)
  out_path   <- input$out_path
  if (is.null(out_path) || !nzchar(out_path)) {
    out_path <- file.path(tempdir(), sprintf("SimulatedProviders_%d.rda", seed))
  }
  if (m < 2) stop("n_providers must be at least 2")
  if (n_per < 2) stop("n_per_provider must be at least 2")
  if (event_rate <= 0 || event_rate >= 1) stop("event_rate must be strictly between 0 and 1")

  set.seed(seed)
  # Unequal cluster sizes: provider volume varies a lot in practice, and equal
  # sizes would hide exactly the small-provider separation Firth is for.
  sizes <- pmax(2L, rpois(m, n_per))
  n <- sum(sizes)
  ID <- rep(sprintf("P%03d", seq_len(m)), sizes)

  Z <- matrix(rnorm(n * p), nrow = n, ncol = p)
  colnames(Z) <- paste0("x", seq_len(p))
  beta <- round(seq(0.5, -0.5, length.out = p), 3)

  gamma <- rnorm(m, mean = 0, sd = gamma_sd)
  # Shift the provider effects so the marginal event rate lands on target.
  offset <- stats::uniroot(
    function(o) mean(plogis(rep(gamma, sizes) + o + Z %*% beta)) - event_rate,
    interval = c(-20, 20))$root
  gamma <- gamma + offset

  eta <- rep(gamma, sizes) + as.numeric(Z %*% beta)
  Y <- rbinom(n, 1, plogis(eta))
  names(gamma) <- sprintf("P%03d", seq_len(m))
  names(beta) <- colnames(Z)

  obj_name <- tools::file_path_sans_ext(basename(out_path))
  dat <- list(Y = Y, Z = Z, ID = ID,
              truth = list(beta = beta, gamma = gamma, seed = seed))
  assign(obj_name, dat)
  dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
  save(list = obj_name, file = out_path)

  zero_event <- sum(tapply(Y, ID, sum) == 0)
  list(status = "ok",
       data_path = normalizePath(out_path),
       object_name = obj_name,
       y_expr = paste0(obj_name, "$Y"),
       z_expr = paste0(obj_name, "$Z"),
       id_expr = paste0(obj_name, "$ID"),
       n_obs = n,
       n_providers = m,
       n_covariates = p,
       event_rate = round(mean(Y), 4),
       providers_with_no_events = zero_event,
       true_beta = as.list(beta),
       seed = seed)
})
write_result(result, a$output)
