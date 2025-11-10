# ============================================================================
# Test Script: Compare MyKmeans (R) and MyKmeans_c (Rcpp) Results
# ============================================================================

library(Rcpp)
library(RcppArmadillo)

# Source the Rcpp code
sourceCpp("./src/kmeans.cpp")  # 替换为你的cpp文件名

# Load or define the original R function
source("./R/Kmeans_wrapper.R")  # 或者直接粘贴MyKmeans函数定义

# ============================================================================
# Test Function
# ============================================================================

test_kmeans_equivalence <- function(X, K, M = NULL, numIter = 100, 
                                    test_name = "Test") {
  cat("\n", rep("=", 60), "\n", sep = "")
  cat(test_name, "\n")
  cat(rep("=", 60), "\n", sep = "")
  
  # Run R version
  cat("Running R version...\n")
  time_r <- system.time({
    Y_r <- MyKmeans(X, K, M, numIter)
  })
  
  # Run Rcpp version
  # Need to handle M = NULL case for Rcpp
  cat("Running Rcpp version...\n")
  if (is.null(M)) {
    # If M is NULL, initialize it the same way for both versions
    set.seed(123)
    idx <- sample.int(nrow(X), K, replace = FALSE)
    M <- X[idx, , drop = FALSE]
    
    # Re-run R version with this M for fair comparison
    Y_r <- MyKmeans(X, K, M, numIter)
  }
  
  time_cpp <- system.time({
    Y_cpp <- MyKmeans_c(X, K, M, numIter)
  })
  
  # Compare results
  cat("\nResults Comparison:\n")
  cat("R version time:   ", time_r[3], "seconds\n")
  cat("Rcpp version time:", time_cpp[3], "seconds\n")
  cat("Speedup:          ", round(time_r[3] / time_cpp[3], 2), "x\n")
  
  # Check if results are identical
  identical_results <- identical(Y_r, as.integer(Y_cpp))
  cat("\nIdentical results:", identical_results, "\n")
  
  if (!identical_results) {
    # Check if clustering is equivalent (permutation of labels)
    cat("Checking if results are equivalent (up to label permutation)...\n")
    
    # Check if cluster sizes match
    table_r <- table(Y_r)
    table_cpp <- table(Y_cpp)
    cat("Cluster sizes (R):  ", sort(as.vector(table_r)), "\n")
    cat("Cluster sizes (Cpp):", sort(as.vector(table_cpp)), "\n")
    
    # Check adjusted Rand index (requires package)
    if (requireNamespace("mclust", quietly = TRUE)) {
      ari <- mclust::adjustedRandIndex(Y_r, Y_cpp)
      cat("Adjusted Rand Index:", ari, "\n")
      if (ari > 0.99) {
        cat("Results are equivalent (same clustering, different labels)\n")
      }
    }
    
    # Show confusion matrix
    cat("\nConfusion Matrix:\n")
    print(table(R_version = Y_r, Rcpp_version = Y_cpp))
  }
  
  cat("\n", rep("-", 60), "\n", sep = "")
  
  return(list(
    identical = identical_results,
    Y_r = Y_r,
    Y_cpp = Y_cpp,
    time_r = time_r[3],
    time_cpp = time_cpp[3]
  ))
}

# ============================================================================
# Test Cases
# ============================================================================

cat("\n")
cat("##########################################################\n")
cat("#  Testing K-means: R vs Rcpp Implementation            #\n")
cat("##########################################################\n")

# Test 1: Simple 2D data, 2 clusters
cat("\n>>> Test 1: Simple 2D data with 2 well-separated clusters\n")
set.seed(123)
X1 <- matrix(rnorm(100, mean = 0, sd = 0.3), ncol = 2)
X2 <- matrix(rnorm(100, mean = 3, sd = 0.3), ncol = 2)
X <- rbind(X1, X2)

# Initialize M explicitly for reproducibility
set.seed(456)
idx <- sample(nrow(X), 2)
M_init <- X[idx, ]

test1 <- test_kmeans_equivalence(X, K = 2, M = M_init, numIter = 100,
                                 test_name = "Test 1: 2D, 2 clusters")

# Test 2: 3 clusters in 2D
cat("\n>>> Test 2: 2D data with 3 clusters\n")
set.seed(234)
X1 <- matrix(rnorm(60, mean = 0, sd = 0.3), ncol = 2)
X2 <- matrix(rnorm(60, mean = 3, sd = 0.3), ncol = 2)
X3 <- matrix(rnorm(60, mean = c(1.5, 3), sd = 0.3), ncol = 2)
X <- rbind(X1, X2, X3)

set.seed(567)
idx <- sample(nrow(X), 3)
M_init <- X[idx, ]

test2 <- test_kmeans_equivalence(X, K = 3, M = M_init, numIter = 100,
                                 test_name = "Test 2: 2D, 3 clusters")

# Test 3: Higher dimensional data (5D)
cat("\n>>> Test 3: 5D data with 4 clusters\n")
set.seed(345)
n_per_cluster <- 50
X1 <- matrix(rnorm(n_per_cluster * 5, mean = 0, sd = 0.5), ncol = 5)
X2 <- matrix(rnorm(n_per_cluster * 5, mean = 2, sd = 0.5), ncol = 5)
X3 <- matrix(rnorm(n_per_cluster * 5, mean = -2, sd = 0.5), ncol = 5)
X4 <- matrix(rnorm(n_per_cluster * 5, mean = 4, sd = 0.5), ncol = 5)
X <- rbind(X1, X2, X3, X4)

set.seed(678)
idx <- sample(nrow(X), 4)
M_init <- X[idx, ]

test3 <- test_kmeans_equivalence(X, K = 4, M = M_init, numIter = 100,
                                 test_name = "Test 3: 5D, 4 clusters")

# Test 4: Larger dataset for speed comparison
cat("\n>>> Test 4: Larger dataset (1000 points, 10D, 5 clusters)\n")
set.seed(456)
n_per_cluster <- 200
K <- 5
p <- 10
X_list <- lapply(1:K, function(k) {
  matrix(rnorm(n_per_cluster * p, mean = k * 3, sd = 0.8), ncol = p)
})
X <- do.call(rbind, X_list)

set.seed(789)
idx <- sample(nrow(X), K)
M_init <- X[idx, ]

test4 <- test_kmeans_equivalence(X, K = K, M = M_init, numIter = 100,
                                 test_name = "Test 4: Large dataset")

# Test 5: Test with limited iterations
cat("\n>>> Test 5: Limited iterations (numIter = 5)\n")
set.seed(567)
X1 <- matrix(rnorm(100, mean = 0, sd = 0.5), ncol = 2)
X2 <- matrix(rnorm(100, mean = 2, sd = 0.5), ncol = 2)
X <- rbind(X1, X2)

set.seed(890)
idx <- sample(nrow(X), 2)
M_init <- X[idx, ]

test5 <- test_kmeans_equivalence(X, K = 2, M = M_init, numIter = 5,
                                 test_name = "Test 5: Limited iterations")

# ============================================================================
# Summary
# ============================================================================

cat("\n")
cat("##########################################################\n")
cat("#  Summary of All Tests                                 #\n")
cat("##########################################################\n\n")

all_tests <- list(test1, test2, test3, test4, test5)
test_names <- c("Test 1", "Test 2", "Test 3", "Test 4", "Test 5")

results_df <- data.frame(
  Test = test_names,
  Identical = sapply(all_tests, function(x) x$identical),
  Time_R = sapply(all_tests, function(x) round(x$time_r, 4)),
  Time_Rcpp = sapply(all_tests, function(x) round(x$time_cpp, 4)),
  Speedup = sapply(all_tests, function(x) round(x$time_r / x$time_cpp, 2))
)

print(results_df)

cat("\n")
if (all(results_df$Identical)) {
  cat("✓ SUCCESS: All tests passed! R and Rcpp versions produce identical results.\n")
} else {
  cat("✗ WARNING: Some tests show different results. Check individual test outputs above.\n")
}

cat("\nAverage speedup:", round(mean(results_df$Speedup), 2), "x\n")