#' Multinomial Logistic Regression
#'
#' Fits a multi-class (softmax) logistic regression by (damped) Newton/gradient-style
#' updates implemented in C++ via RcppArmadillo.
#' 
#' @param X Numeric matrix \eqn{n \times p}. The first column **must** be all 1s (intercept).
#' @param y Integer vector of length \eqn{n} with class labels in \code{0, ..., K-1}.
#' @param numIter Positive integer: number of fixed iterations (default \code{50}).
#' @param eta Positive numeric step size (default \code{0.1}).
#' @param lambda Non-negative numeric L2 penalty (default \code{1}).
#' @param beta_init Optional numeric matrix \eqn{p \times K} of initial coefficients. If \code{NULL}, initialized to zeros.
#'
#' @return A list with:
#' \item{beta}{Numeric matrix \eqn{p \times K} of coefficients (including intercept row).}
#' \item{objective}{Numeric vector of length \code{numIter + 1} with objective values per iteration (including start).}
#'
#' @export
#'
#' @examples
#' set.seed(1)
#' n <- 100; p <- 4; K <- 3
#' X <- cbind(1, matrix(rnorm(n*(p-1)), n, p-1))
#' beta_true <- matrix(rnorm(p*K, sd = 0.5), p, K)
#' S <- X %*% beta_true
#' Smax <- apply(S, 1, max)
#' P <- exp(S - Smax); P <- P / rowSums(P)
#' y <- apply(P, 1, function(pr) sample.int(K, 1, prob = pr)) - 1L
#' fit <- LRMultiClass(X, y, numIter = 20, eta = 0.2, lambda = 1)
#' str(fit)
LRMultiClass <- function(X, y, beta_init = NULL, numIter = 50, eta = 0.1, lambda = 1){
  
  # Compatibility checks from HW3 and initialization of beta_init
  # Check that the first column of X are 1s, if not - display appropriate message and stop execution.
  if (!all(X[,1] == 1)) {
    stop("The first column of X must be all 1s.")
  }
  # Check for compatibility of dimensions between X and Y
  n <- nrow(X)
  p <- ncol(X)
  K <- length(unique(y))
  if (length(y) != n) {
    stop("Length of y must match the number of rows in X.")
  }
  if (!all(y %in% 0:(K-1))) {
    stop("y must only contain class labels in {0, ..., K-1}.")
  }
  # Check eta is positive
  if (eta <= 0) {
    stop("eta must be a positive scalar.")
  }
  # Check lambda is non-negative
  if (lambda < 0) {
    stop("lambda must be a non-negative scalar.")
  }
  # Check whether beta_init is NULL. If NULL, initialize beta with p x K matrix of zeroes. If not NULL, check for compatibility of dimensions with what has been already supplied.
  if (is.null(beta_init)) {
    beta_init <- matrix(0, nrow = p, ncol = K)
  } else {
    if (!all(dim(beta_init) == c(p, K))) {
      stop("beta_init must have dimensions p x K.")
    }
  }
  
  # Call C++ LRMultiClass_c function to implement the algorithm
  out = LRMultiClass_c(X, y, beta_init, numIter, eta, lambda)
  
  # Return the class assignments
  return(out)
}