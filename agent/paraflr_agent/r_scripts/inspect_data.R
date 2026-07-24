#!/usr/bin/env Rscript
# inspect_data.R — describe an R/CSV data file, and guess the paraflr mapping.
#
# input.json : {"data_path": "..."}
# output.json: {"status":"ok", "structure": {...}, "mapping": {...}|null}
#
# Only metadata (class, dim, length) is reported for `structure` — values are
# never returned, so this is safe on very large files. `mapping` is the extra
# piece the agent harness relies on: a best guess at which fields are the
# binary outcome, the covariate matrix, and the provider id, expressed as R
# expressions the tool scripts can evaluate verbatim. Guessing here (in R,
# deterministically) is what keeps the language model out of the business of
# inventing field names.

.script_dir <- function() {
  f <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  # Rscript encodes spaces in --file= as "~+~" (paths with spaces hit this).
  if (length(f)) dirname(gsub("~\\+~", " ", sub("^--file=", "", f[1]))) else getwd()
}
source(file.path(.script_dir(), "_common.R"))

describe <- function(obj) {
  if (is.matrix(obj)) {
    sprintf("matrix %dx%d (%s)", nrow(obj), ncol(obj), typeof(obj))
  } else if (is.data.frame(obj)) {
    list(".kind" = sprintf("data.frame %dx%d", nrow(obj), ncol(obj)),
         ".columns" = as.list(vapply(obj, function(x) class(x)[1], character(1))))
  } else if (is.list(obj) && !is.null(names(obj)) && all(nzchar(names(obj)))) {
    lapply(obj, describe)
  } else if (is.list(obj)) {
    sprintf("unnamed list (length %d)", length(obj))
  } else if (is.factor(obj)) {
    sprintf("factor (length %d, %d levels)", length(obj), nlevels(obj))
  } else if (is.numeric(obj)) {
    if (all(obj %in% c(0, 1))) {
      sprintf("numeric 0/1 (length %d, %d ones)", length(obj), sum(obj == 1))
    } else {
      sprintf("numeric (length %d)", length(obj))
    }
  } else if (is.character(obj)) {
    sprintf("character (length %d, %d unique)", length(obj),
            length(unique(obj)))
  } else if (is.logical(obj)) {
    sprintf("logical (length %d)", length(obj))
  } else {
    class(obj)[1]
  }
}

# --- mapping guess -----------------------------------------------------------
#
# Walk one candidate container (a list, a data.frame, or the top-level
# environment) and look for the three fields logis_firth() needs:
#   y  : numeric/logical vector, values in {0,1}
#   z  : matrix or data.frame of covariates, nrow == length(y)
#   id  : vector of the same length with far fewer unique values than records
# Returns NULL unless all three are found in one container — a partial guess
# would be worse than none, because the agent would fill the gap by inventing
# a name.

is_binary_vec <- function(x) {
  (is.numeric(x) || is.logical(x)) && !is.matrix(x) && length(x) > 1 &&
    all(!is.na(x)) && all(as.numeric(x) %in% c(0, 1))
}

is_id_vec <- function(x, n) {
  if (is.matrix(x) || is.list(x)) return(FALSE)
  if (length(x) != n) return(FALSE)
  if (!(is.character(x) || is.factor(x) || is.numeric(x) || is.integer(x))) {
    return(FALSE)
  }
  u <- length(unique(x))
  u > 1 && u < n / 2
}

guess_in_container <- function(obj, prefix) {
  if (!(is.list(obj) || is.data.frame(obj))) return(NULL)
  nms <- names(obj)
  if (is.null(nms) || !all(nzchar(nms))) return(NULL)

  path <- function(nm) paste0(prefix, "$", nm)

  # Outcome: prefer a field literally named y / status / event / outcome.
  y_cands <- nms[vapply(nms, function(nm) is_binary_vec(obj[[nm]]), logical(1))]
  if (!length(y_cands)) return(NULL)
  preferred <- c("y", "Y", "status", "event", "outcome", "died", "death")
  y_nm <- if (any(y_cands %in% preferred)) {
    y_cands[y_cands %in% preferred][1]
  } else {
    y_cands[1]
  }
  n <- length(obj[[y_nm]])

  # Covariates: a matrix / data.frame with matching row count.
  z_cands <- nms[vapply(nms, function(nm) {
    x <- obj[[nm]]
    (is.matrix(x) || is.data.frame(x)) && nrow(x) == n
  }, logical(1))]
  if (!length(z_cands)) return(NULL)
  z_pref <- c("z", "Z", "x", "X", "covariates")
  z_nm <- if (any(z_cands %in% z_pref)) z_cands[z_cands %in% z_pref][1] else z_cands[1]

  # Provider id.
  id_cands <- nms[vapply(nms, function(nm) {
    nm != y_nm && is_id_vec(obj[[nm]], n)
  }, logical(1))]
  if (!length(id_cands)) return(NULL)
  id_pref <- c("ID", "id", "provider", "provider_id", "prov", "facility",
               "center", "cluster", "stratum")
  id_nm <- if (any(id_cands %in% id_pref)) {
    id_cands[id_cands %in% id_pref][1]
  } else {
    id_cands[1]
  }

  z <- obj[[z_nm]]
  list(y_expr = path(y_nm), z_expr = path(z_nm), id_expr = path(id_nm),
       n_obs = n,
       n_covariates = ncol(z),
       covariate_names = if (!is.null(colnames(z))) as.list(colnames(z)) else NULL,
       n_providers = length(unique(obj[[id_nm]])),
       event_rate = round(mean(as.numeric(obj[[y_nm]])), 4))
}

guess_mapping <- function(e) {
  top <- as.list(e)
  # Top-level environment itself (each object a separate field), then each
  # named list / data.frame one level down.
  for (nm in names(top)) {
    g <- guess_in_container(top[[nm]], nm)
    if (!is.null(g)) return(g)
  }
  # Objects sitting side by side at top level: Y, Z, ID as separate objects.
  fake <- guess_in_container(top, "")
  if (!is.null(fake)) {
    fake$y_expr  <- sub("^\\$", "", fake$y_expr)
    fake$z_expr  <- sub("^\\$", "", fake$z_expr)
    fake$id_expr <- sub("^\\$", "", fake$id_expr)
    return(fake)
  }
  NULL
}

a <- parse_args()
result <- with_error_payload("inspect_data.R", {
  input <- read_input(a$input)
  e <- load_data_env(input$data_path)
  structure_desc <- lapply(as.list(e), describe)
  list(status = "ok",
       data_path = input$data_path,
       top_level_names = names(structure_desc),
       structure = structure_desc,
       mapping = guess_mapping(e))
})
write_result(result, a$output)
