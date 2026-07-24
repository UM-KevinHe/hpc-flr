test_gamma.single <- function(object, methods = c("wald", "score", "lrt"), null = "median",
                              alpha = 0.05, firth = FALSE) {
  methods <- match.arg(methods)
  data <- object$data
  Y <- data[[object$char_list$Y.char]]
  Z <- as.matrix(data[, object$char_list$Z.char, drop = FALSE])
  prov <- as.character(data[[object$char_list$prov.char]])
  n.prov <- rle(prov)$lengths

  gamma <- object$gamma
  beta <- object$beta
  gamma1 <- gamma[1]
  gamma.null <- if (identical(null, "median")) median(gamma)
                else if (is.numeric(null)) null[1]
                else stop("Argument 'null' NOT as required!", call. = FALSE)

  res <- matrix(NA, nrow = 1, ncol = 4)
  colnames(res) <- c("gamma_est", "stats", "p", "flag")
  res[1, 1] <- gamma1

  if (methods == "wald") {
    gamma_obs <- rep(gamma, n.prov)
    probs <- as.numeric(plogis(gamma_obs + Z %*% beta))
    se_gamma <- wald_gamma(Z, probs, n.prov, 1)
    if (any(is.na(se_gamma))) {
      res[1, 2] <- NA
      res[1, 3] <- NA
    } else {
      wald_stat <- (gamma1 - gamma.null) / se_gamma
      res[1, 2] <- wald_stat
      res[1, 3] <- 2 * pnorm(-abs(wald_stat))
    }
  } else if (methods == "score") {
    idx <- seq_len(n.prov[1])
    prob_null <- plogis(gamma.null + Z[idx, , drop = FALSE] %*% beta)
    zscore <- sum(Y[idx] - prob_null) / sqrt(sum(prob_null * (1 - prob_null)))
    res[1, 2] <- zscore
    res[1, 3] <- 2 * pnorm(-abs(zscore))
  } else if (methods == "lrt") {
    gamma_test <- gamma
    gamma_test[1] <- gamma.null
    if (firth) {
      full_lkd <- Loglkd_firth(Y = Y, Z = Z, n_prov = n.prov, gamma = gamma, beta = beta)
      null_lkd <- Loglkd_firth(Y = Y, Z = Z, n_prov = n.prov, gamma = gamma_test, beta = beta)
    } else {
      Z_beta <- Z %*% beta
      full_lkd <- Loglkd(Y = Y, Z_beta = Z_beta, gamma_obs = rep(gamma, n.prov))
      null_lkd <- Loglkd(Y = Y, Z_beta = Z_beta, gamma_obs = rep(gamma_test, n.prov))
    }
    LRT_stat <- -2 * (null_lkd - full_lkd)
    res[1, 2] <- LRT_stat
    res[1, 3] <- pchisq(LRT_stat, df = 1, lower.tail = FALSE)
  }

  if (is.na(res[1, 3])) {
    res[1, 4] <- NA
  } else if (res[1, 3] < alpha) {
    res[1, 4] <- if (methods == "lrt") ifelse(gamma1 - gamma.null > 0, 1, -1)
                 else ifelse(res[1, 2] > 0, 1, -1)
  } else {
    res[1, 4] <- 0
  }
  res
}
