#!/usr/bin/env Rscript
# Fit paraflr::logis_firth on a CSV and emit gamma/beta as JSON.
# Used by parity_with_r.py to compare the R and Python builds.
#   Rscript _fit_in_r.R <in.csv> <out.json>
# Optional: set PARAFLR_RLIB to a library path holding the paraflr install.
rlib <- Sys.getenv("PARAFLR_RLIB", "")
if (nzchar(rlib)) .libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages({ library(paraflr); library(jsonlite) })

args <- commandArgs(trailingOnly = TRUE)
d  <- read.csv(args[1])
Y  <- d$y
ID <- d$id
Z  <- as.matrix(d[, grep("^z", names(d)), drop = FALSE])

fit <- logis_firth(Y, Z, ID, threads = 1)

out <- list(gamma = as.list(fit$gamma),
            beta  = as.list(fit$beta),
            neg2Loglkd = fit$neg2Loglkd)
write(toJSON(out, auto_unbox = TRUE, digits = 15), args[2])
