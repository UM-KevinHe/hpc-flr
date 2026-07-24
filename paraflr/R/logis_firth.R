logis_firth <- function(Y, Z, ID, cutoff = 0, max.iter = 10000, tol = 1e-5,
                        bound = 10, backtrack = FALSE, threads = 1, message = FALSE) {
  Z <- as.matrix(Z)
  Z.char <- colnames(Z)
  if (is.null(Z.char)) {
    Z.char <- paste0("Z", seq_len(ncol(Z)))
    colnames(Z) <- Z.char
  }
  Y.char <- "Y"
  prov.char <- "ID"

  data <- data.frame(Y = Y, ID = ID, check.names = FALSE)
  data[, Z.char] <- Z
  data <- data[order(data[[prov.char]]), ]

  blocks <- rle(as.character(data[[prov.char]]))
  size.long <- rep(blocks$lengths, blocks$lengths)
  data <- data[size.long >= cutoff, ]

  blocks <- rle(as.character(data[[prov.char]]))
  n.prov <- blocks$lengths
  prov.ids <- blocks$values

  Zmat <- as.matrix(data[, Z.char, drop = FALSE])
  m <- length(n.prov)
  n.obs <- nrow(data)
  gamma <- rep(log(mean(data[[Y.char]]) / (1 - mean(data[[Y.char]]))), m)
  beta <- rep(0, ncol(Zmat))

  fit <- logis_firth_prov(as.matrix(data[[Y.char]]), Zmat, n.prov, gamma, beta,
                          n.obs, m, threads = threads, tol = tol, max_iter = max.iter,
                          bound = bound, message = message, backtrack = backtrack)

  gamma <- as.numeric(fit$gamma)
  beta <- as.numeric(fit$beta)
  names(gamma) <- prov.ids
  names(beta) <- Z.char

  gamma.obs <- rep(gamma, n.prov)
  eta <- gamma.obs + as.numeric(Zmat %*% beta)
  neg2Loglkd <- -2 * sum(eta * data[[Y.char]] - log(1 + exp(eta)))

  char_list <- list(Y.char = Y.char, prov.char = prov.char, Z.char = Z.char)
  structure(list(data = data, char_list = char_list, beta = beta, gamma = gamma,
                 neg2Loglkd = neg2Loglkd),
            class = "logis_firth")
}
