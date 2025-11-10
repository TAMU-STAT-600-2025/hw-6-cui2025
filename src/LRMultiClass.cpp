// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

// For simplicity, no test data, only training data, and no error calculation.
// X - n x p data matrix
// y - n length vector of classes, from 0 to K-1
// numIter - number of iterations, default 50
// eta - damping parameter, default 0.1
// lambda - ridge parameter, default 1
// beta_init - p x K matrix of starting beta values (always supplied in right format)

// stable row-wise softmax: subtract row max before exp
static inline arma::mat row_softmax(const arma::mat& S) {
  arma::mat Z = S;                         // n x K
  arma::colvec rmax = arma::max(Z, 1);     // n x 1
  Z.each_col() -= rmax;
  Z = arma::exp(Z);
  arma::colvec rsum = arma::sum(Z, 1);     // n x 1
  Z.each_col() /= rsum;
  return Z;                                // rows sum to 1
}

// compute objective: -sum log p(y_i) + (lambda/2) * ||beta||_F^2
static inline double objective_fn(const arma::mat& X,
                                  const arma::uvec& y,
                                  const arma::mat& beta,
                                  double lambda) {
  arma::mat S = X * beta;                  // n x K
  arma::mat P = row_softmax(S);            // n x K
  
  // gather log-prob of true class for each i
  // create linear indices for P(i, y[i])
  const arma::uword n = P.n_rows;
  const arma::uword K = P.n_cols;
  arma::uvec col_idx = y;                  // length n, in [0..K-1]
  arma::uvec row_idx = arma::regspace<arma::uvec>(0, n - 1);
  arma::uvec lin_idx = col_idx * n + row_idx; // column-major in Armadillo
  
  // Armadillo matrices are column-major; vectorise by column, then index
  arma::vec Pvec = arma::vectorise(P);     // length n*K
  double nll = -arma::sum(arma::log(Pvec.elem(lin_idx)));
  
  double penalty = 0.5 * lambda * arma::accu(arma::square(beta));
  return nll + penalty;
}

// [[Rcpp::export]]
Rcpp::List LRMultiClass_c(const arma::mat& X, const arma::uvec& y, const arma::mat& beta_init,
                               int numIter = 50, double eta = 0.1, double lambda = 1){
    // All input is assumed to be correct
    
    // Initialize some parameters
    int K = max(y) + 1; // number of classes
    int p = X.n_cols;
    int n = X.n_rows;
    arma::mat beta = beta_init; // to store betas and be able to change them if needed
    arma::vec objective(numIter + 1); // to store objective values
    
    // Initialize anything else that you may need
    objective(0) = objective_fn(X, y, beta, lambda);
    
    // Pre-compute identity matrix for Hessian (avoid repeated allocation) (by Cui)
    const arma::mat I_p = arma::eye<arma::mat>(p, p); 
    
    // Newton's method cycle - implement the update EXACTLY numIter iterations
    for (int iter = 0; iter < numIter; ++iter) {
      // Step 1: current probabilities under beta (by Cui)
      arma::mat S = X * beta;                    // n x K
      arma::mat P = row_softmax(S);              // n x K
      
      // Step 2: Perform class-wise Newton updates (by Cui)
      // Each class k is updated independently using its own gradient and Hessian (by Cui)
      for (arma::uword k = 0; k < K; ++k) {
        // pk = P[, k]
        // Extract probabilities for class k (by Cui)
        arma::vec pk = P.col(k);               // length n
        
        // w = pk * (1 - pk)
        // Compute weights for weighted least squares: w_i = p_k(i) * (1 - p_k(i))  (by Cui)
        arma::vec w = pk % (1.0 - pk);         // length n
        
        // indicator for class k: y == k (as double vector)
        arma::vec yeqk = arma::conv_to<arma::vec>::from(y == k);
        
        // This is the derivative of the objective w.r.t. beta_k (by Cui)
        // gradient: gk = X^T (pk - I[y==k]) + lambda * beta_k
        arma::vec gk = X.t() * (pk - yeqk) + lambda * beta.col(k); // p x 1
        
        // Hessian: Hk = X^T diag(w) X + lambda * I_p
        // compute Xw = diag(w) * X by scaling rows of X
        arma::mat Xw = X.each_col() % w;       // n x p
        arma::mat Hk = X.t() * Xw + lambda * I_p; // p x p
        
        // Newton step: solve(Hk, gk)
        arma::vec step = arma::solve(Hk, gk, arma::solve_opts::fast);
        
        // damped update
        beta.col(k) -= eta * step;
      }
      
      // Step 3: Record objective value after this iteration (by Cui)
      objective(iter + 1) = objective_fn(X, y, beta, lambda);
    }
    
    // Create named list with betas and objective values
    return Rcpp::List::create(Rcpp::Named("beta") = beta,
                              Rcpp::Named("objective") = objective);
}
