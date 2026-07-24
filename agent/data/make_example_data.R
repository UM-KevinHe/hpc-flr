#!/usr/bin/env Rscript
# make_example_data.R — build the two bundled example datasets.
#
#   Rscript agent/data/make_example_data.R
#
# The example data is simulated. Both files are produced by the agent's own
# simulate_provider_data tool through the same JSON handshake the agent uses.
#
#   ExampleProviders.rda      50 providers, ~80 records each, 20% event rate
#                             — the ordinary profiling case.
#   ExampleProviders_rare.rda 40 small providers, ~25 records each, 3% event
#                             rate — several providers see no events at all,
#                             which is where the Firth correction earns its
#                             keep and where Wald and score disagree.

suppressPackageStartupMessages(library(jsonlite))

HERE <- (function() {
  f <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(f)) dirname(normalizePath(gsub("~\\+~", " ", sub("^--file=", "", f[1])))) else getwd()
})()
R_SCRIPTS <- file.path(dirname(HERE), "paraflr_agent", "r_scripts")

run_tool <- function(tool, args) {
  in_file  <- tempfile(fileext = ".in.json")
  out_file <- tempfile(fileext = ".out.json")
  on.exit({ unlink(in_file); unlink(out_file) }, add = TRUE)
  writeLines(toJSON(args, auto_unbox = TRUE, null = "null"), in_file)
  cmd <- paste(shQuote(file.path(R.home("bin"), "Rscript")),
               "--no-save --no-restore --no-init-file",
               shQuote(file.path(R_SCRIPTS, paste0(tool, ".R"))),
               shQuote(in_file), shQuote(out_file))
  status <- system(cmd)
  if (!file.exists(out_file)) {
    stop(sprintf("%s.R produced no output (status=%d)", tool, status))
  }
  fromJSON(out_file)
}

specs <- list(
  list(out_path = file.path(HERE, "ExampleProviders.rda"),
       n_providers = 50L, n_per_provider = 80L, n_covariates = 5L,
       event_rate = 0.2, gamma_sd = 0.5, seed = 2026L),
  list(out_path = file.path(HERE, "ExampleProviders_rare.rda"),
       n_providers = 40L, n_per_provider = 25L, n_covariates = 4L,
       event_rate = 0.03, gamma_sd = 0.8, seed = 7L)
)

for (spec in specs) {
  res <- run_tool("simulate_provider_data", spec)
  if (!identical(res$status, "ok")) {
    stop(sprintf("simulation failed: %s", res$message))
  }
  cat(sprintf(
    "%-28s n=%d, providers=%d, p=%d, event rate=%.3f, providers with no events=%d\n",
    basename(res$data_path), res$n_obs, res$n_providers, res$n_covariates,
    res$event_rate, res$providers_with_no_events))
}
