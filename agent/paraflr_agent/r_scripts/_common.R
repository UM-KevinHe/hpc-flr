#!/usr/bin/env Rscript
# _common.R — shared plumbing for every paraflr agent R script.
#
# Each tool script is called as
#   Rscript <tool>.R <input.json> <output.json>
# and communicates only through those two JSON files, so the Python side
# never has to parse R console output.
#
# The six tool scripts share this loader instead of each duplicating it.

suppressPackageStartupMessages({
  library(jsonlite)
})

# --- argument handling -------------------------------------------------------

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 2) {
    stop("Usage: Rscript <tool>.R <input.json> <output.json>")
  }
  list(input = args[1], output = args[2])
}

read_input <- function(path) {
  fromJSON(path, simplifyVector = FALSE)
}

write_result <- function(result, path) {
  writeLines(
    toJSON(result, auto_unbox = TRUE, matrix = "rowmajor", na = "null",
           null = "null", digits = NA, pretty = TRUE),
    con = path
  )
}

# Wrap a tool body so that any R condition becomes a structured error
# payload instead of a traceback on stderr.
with_error_payload <- function(where, expr) {
  tryCatch(expr, error = function(err) {
    list(status = "error",
         message = conditionMessage(err),
         class = class(err)[1],
         where = where)
  })
}

# --- data loading ------------------------------------------------------------

# Load a data file into a fresh environment. .rda/.RData may carry several
# objects; .rds and .csv carry one, named after the file so that *_expr
# arguments have something to refer to.
load_data_env <- function(data_path) {
  if (is.null(data_path) || !nzchar(data_path)) stop("data_path is required")
  if (!file.exists(data_path)) {
    stop(sprintf("File not found: %s", data_path))
  }
  ext <- tolower(tools::file_ext(data_path))
  e <- new.env()
  if (ext %in% c("rda", "rdata")) {
    load(data_path, envir = e)
  } else if (ext == "rds") {
    assign(tools::file_path_sans_ext(basename(data_path)),
           readRDS(data_path), envir = e)
  } else if (ext == "csv") {
    assign(tools::file_path_sans_ext(basename(data_path)),
           utils::read.csv(data_path, stringsAsFactors = FALSE), envir = e)
  } else {
    stop(sprintf("Unsupported file extension: .%s (supported: .rda .RData .rds .csv)",
                 ext))
  }
  e
}

# Evaluate an R expression string inside `env`; NULL for empty input.
eval_in <- function(expr_str, env) {
  if (is.null(expr_str) || !is.character(expr_str) || !nzchar(expr_str)) {
    return(NULL)
  }
  eval(parse(text = expr_str), envir = env)
}

# Resolve the three fields every paraflr tool needs.
resolve_flr_fields <- function(input, e) {
  Y <- eval_in(input$y_expr, e)
  if (is.null(Y)) stop("y_expr is required and must resolve to a binary vector")
  Y <- as.numeric(Y)
  if (!all(Y %in% c(0, 1))) {
    stop("y_expr must resolve to a 0/1 binary outcome vector")
  }

  Z <- eval_in(input$z_expr, e)
  if (is.null(Z)) stop("z_expr is required and must resolve to a matrix or data.frame")
  Z <- as.matrix(Z)
  storage.mode(Z) <- "double"

  ID <- eval_in(input$id_expr, e)
  if (is.null(ID)) stop("id_expr is required and must resolve to a provider identifier")
  ID <- as.vector(ID)

  if (length(Y) != nrow(Z) || length(Y) != length(ID)) {
    stop(sprintf(
      "Field lengths differ: y has %d, z has %d rows, id has %d. The three must describe the same records.",
      length(Y), nrow(Z), length(ID)))
  }
  list(Y = Y, Z = Z, ID = ID)
}

# Optional fitting controls, with logis_firth()'s own defaults.
fit_controls <- function(input) {
  num <- function(x, default) if (is.null(x)) default else as.numeric(x)
  list(
    threads   = if (is.null(input$threads)) 1L else as.integer(input$threads),
    cutoff    = num(input$cutoff, 0),
    tol       = num(input$tol, 1e-5),
    max.iter  = if (is.null(input$max_iter)) 10000L else as.integer(input$max_iter),
    bound     = num(input$bound, 10),
    backtrack = if (is.null(input$backtrack)) FALSE else as.logical(input$backtrack)
  )
}

run_fit <- function(fields, ctl) {
  paraflr::logis_firth(
    Y = fields$Y, Z = fields$Z, ID = fields$ID,
    cutoff = ctl$cutoff, max.iter = ctl$max.iter, tol = ctl$tol,
    bound = ctl$bound, backtrack = ctl$backtrack, threads = ctl$threads,
    message = FALSE
  )
}
