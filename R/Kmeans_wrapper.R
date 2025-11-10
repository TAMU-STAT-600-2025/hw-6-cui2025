#' K-means algorithm
#' 
#' the algorithm iteratively divides the points into K clusters/groups such that the points within each group are most similar.
#'
#' @param X - n by p matrix containing n data points to cluster
#' @param K - integer specifying number of clusters
#' @param M - (optional) K by p matrix of cluster centers
#' @param numIter - number of maximal iterations for the algorithm, the default value is 100 
#'
#' @return Explain return # Return the vector of assignments Y
#' @export
#'
#' @examples
#' # Give example
#' X1 <- matrix(rnorm(100, mean = 0, sd = 0.3), ncol = 2)
#' X2 <- matrix(rnorm(100, mean = 3, sd = 0.3), ncol = 2)
#' X  <- rbind(X1, X2)
#' Y <- MyKmeans(X, K = 2)
#' 
MyKmeans <- function(X, K, M = NULL, numIter = 100){
  
  # Check whether M is NULL or not. If NULL, initialize based on K randomly selected points from X. 
  # If not NULL, check for compatibility with X dimensions and K.
  
  # Implement K-means algorithm. 
  # It should stop when either 
  # (i) the centroids don't change from one iteration to the next (exactly the same), or
  # (ii) the maximal number of iterations was reached, or
  # (iii) one of the clusters has disappeared after one of the iterations (in which case the error message is returned)
  
  # check data format and quality, stop if the input data is bad
  # browser()
  X <- as.matrix(X)
  if (!is.numeric(X)) stop("X must be a numeric matrix.")
  n <- nrow(X); p <- ncol(X)
  if (K < 1 || K > n) stop("K must be between 1 and nrow(X).")
  if (!is.null(M)) {
    M <- as.matrix(M)
    if (!is.numeric(M)) stop("M must be a numeric matrix.")
    if (nrow(M) != K) stop("nrow(M) must equal K.")
    if (ncol(M) != p) stop("ncol(M) must equal ncol(X).")
  } else {
    #if M is not given, random select K rows
    idx <- sample.int(n, K, replace = FALSE)
    M <- X[idx, , drop = FALSE]
  }
  
  Y <- integer(n)  # initialize the return 
  x2 <- rowSums(X^2)  # calculate the norm of x2
  
  for (iter in seq_len(numIter)) {
    # assigns each point to the cluster
    # computes the Euclidean distance D(i,k) = ||x_i||^2 + ||mu_k||^2 - 2 * x_i %*% mu_k
    m2 <- rowSums(M^2)                               # K
    D  <- outer(x2, m2, "+") - 2 * (X %*% t(M))      # n x K
    Y_new <- max.col(-D, ties.method = "first")      
    
    # check the number of clusters
    counts <- tabulate(Y_new, nbins = K)            
    if (any(counts == 0L)) {
      stop("A cluster vanished (empty) after an iteration. Try a different initialization or K.")
    }
    
    # recomputes the centroid values
    S <- rowsum(X, group = Y_new, reorder = TRUE)    # K x p
    M_new <- S / counts                               
    
    # check: the centroid values don’t change from one iteration to the next ?
    if (all(M_new == M)) {
      # if yes, then we are done 
      Y <- Y_new
      break
    }
    
    # update parameters
    M <- M_new
    Y <- Y_new
    
    # maximal number of iterations is reached
    if (iter == numIter) break
  }
  
  
  # Return the vector of assignments Y
  return(Y)
}




