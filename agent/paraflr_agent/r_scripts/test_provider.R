#!/usr/bin/env Rscript
# test_provider.R — dispatcher for the test_provider tool:
# paraflr::test_gamma.single() on one provider effect.
#
# input.json : {"data_path", "y_expr", "z_expr", "id_expr",
#               "provider", "method" (wald|score|lrt), "firth", "null",
#               "alpha", plus the fit controls}
# output.json: {"status":"ok", "provider", "method", "gamma_est", "stat",
#               "p_value", "flag", ...}
#
# test_gamma.single() always tests the FIRST provider block of the fitted
# object. logis_firth() sorts records by provider id, so that is the
# smallest id unless we intervene. To test an arbitrary provider we rotate
# the fitted object — its records and its gamma vector together — so the
# requested provider's block comes first. Rotation touches no estimate:
# both objects are simply re-indexed, and the block structure the test
# relies on (contiguous records per provider) is preserved.

.script_dir <- function() {
  f <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  # Rscript encodes spaces in --file= as "~+~" (paths with spaces hit this).
  if (length(f)) dirname(gsub("~\\+~", " ", sub("^--file=", "", f[1]))) else getwd()
}
source(file.path(.script_dir(), "_common.R"))
suppressPackageStartupMessages(library(paraflr))

rotate_to_provider <- function(fit, provider) {
  ids <- names(fit$gamma)
  k <- match(as.character(provider), ids)
  if (is.na(k)) {
    stop(sprintf(
      "Provider '%s' not found. The fitted providers are: %s%s",
      provider, paste(head(ids, 20), collapse = ", "),
      if (length(ids) > 20) sprintf(" ... (%d total)", length(ids)) else ""))
  }
  prov <- as.character(fit$data[[fit$char_list$prov.char]])
  target <- ids[k]
  fit$data <- fit$data[c(which(prov == target), which(prov != target)), ,
                       drop = FALSE]
  fit$gamma <- fit$gamma[c(k, setdiff(seq_along(ids), k))]
  fit
}

a <- parse_args()
result <- with_error_payload("test_provider.R", {
  input  <- read_input(a$input)
  e      <- load_data_env(input$data_path)
  fields <- resolve_flr_fields(input, e)
  ctl    <- fit_controls(input)

  method <- if (is.null(input$method)) "score" else tolower(as.character(input$method))
  if (!method %in% c("wald", "score", "lrt")) {
    stop(sprintf("method must be one of 'wald', 'score', 'lrt' (got '%s')", method))
  }
  firth <- if (is.null(input$firth)) FALSE else as.logical(input$firth)
  alpha <- if (is.null(input$alpha)) 0.05 else as.numeric(input$alpha)
  null_arg <- if (is.null(input$null)) "median" else input$null
  if (is.character(null_arg) && !identical(null_arg, "median")) {
    stop("null must be \"median\" or a number")
  }
  if (!is.character(null_arg)) null_arg <- as.numeric(null_arg)

  fit <- run_fit(fields, ctl)
  provider <- if (is.null(input$provider)) names(fit$gamma)[1] else
    as.character(input$provider)
  fit <- rotate_to_provider(fit, provider)

  res <- test_gamma.single(fit, methods = method, null = null_arg,
                           alpha = alpha, firth = firth)

  flag <- res[1, "flag"]
  verdict <- if (is.na(flag)) "undetermined" else if (flag > 0) {
    "worse than expected (significantly above the null)"
  } else if (flag < 0) {
    "better than expected (significantly below the null)"
  } else {
    "not significantly different from the null"
  }

  list(status = "ok",
       provider = provider,
       method = method,
       firth = firth,
       null = if (is.character(null_arg)) "median" else null_arg,
       null_value = if (is.character(null_arg)) unname(stats::median(fit$gamma))
                    else as.numeric(null_arg),
       alpha = alpha,
       gamma_est = unname(res[1, "gamma_est"]),
       stat = unname(res[1, "stats"]),
       p_value = unname(res[1, "p"]),
       flag = unname(flag),
       verdict = verdict,
       n_obs = nrow(fit$data),
       n_providers = length(fit$gamma),
       provider_n_obs = sum(as.character(
         fit$data[[fit$char_list$prov.char]]) == provider),
       threads = ctl$threads)
})
write_result(result, a$output)
