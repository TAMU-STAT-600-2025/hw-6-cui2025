// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"
// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::uvec MyKmeans_c(const arma::mat& X, int K,
                      const arma::mat& M, int numIter = 100){
  // All input is assumed to be correct
  
  // Initialize some parameters
  int n = X.n_rows;
  int p = X.n_cols;
  arma::uvec Y(n); // to store cluster assignments
  
  // Initialize any additional parameters if needed
  arma::mat centers = M; // copy M since we need to update it
  arma::uvec Y_new(n);
  
  // Pre-compute ||x_i||^2 for all data points
  arma::vec x2 = arma::sum(X % X, 1); // n x 1 vector
  
  // For loop with kmeans algorithm
  for (int iter = 0; iter < numIter; iter++) {
    
    // Step 1: Assign each point to the nearest cluster
    // Compute ||mu_k||^2 for all centers
    arma::vec m2 = arma::sum(centers % centers, 1); // K x 1 vector
    
    // Compute distance matrix D: n x K
    // D(i,k) = ||x_i||^2 + ||mu_k||^2 - 2 * x_i %*% mu_k
    arma::mat D(n, K);
    arma::mat XM = X * centers.t(); // n x K, X %*% t(M)
    
    for (int i = 0; i < n; i++) {
      for (int k = 0; k < K; k++) {
        D(i, k) = x2(i) + m2(k) - 2.0 * XM(i, k);
      }
    }
    
    // Find the cluster with minimum distance for each point
    for (int i = 0; i < n; i++) {
      arma::uword min_idx;
      D.row(i).min(min_idx);
      Y_new(i) = min_idx + 1; // +1 for R's 1-based indexing
    }
    
    // Step 2: Check for empty clusters
    arma::uvec counts = arma::zeros<arma::uvec>(K);
    for (int i = 0; i < n; i++) {
      counts(Y_new(i) - 1)++; // -1 because Y_new is 1-indexed
    }
    
    if (arma::any(counts == 0)) {
      Rcpp::stop("A cluster vanished (empty) after an iteration. Try a different initialization or K.");
    }
    
    // Step 3: Recompute cluster centers
    arma::mat centers_new = arma::zeros<arma::mat>(K, p);
    for (int i = 0; i < n; i++) {
      centers_new.row(Y_new(i) - 1) += X.row(i); // -1 for 0-based indexing
    }
    
    // Divide by counts to get mean
    for (int k = 0; k < K; k++) {
      centers_new.row(k) /= counts(k);
    }
    
    // Step 4: Check for convergence (centers don't change)
    if (arma::approx_equal(centers_new, centers, "absdiff", 1e-10)) {
      Y = Y_new;
      break;
    }
    
    // Update parameters for next iteration
    centers = centers_new;
    Y = Y_new;
  }
  
  // Returns the vector of cluster assignments
  return(Y);
}