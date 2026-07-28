#!/usr/bin/env Rscript

# Standalone R script for running INSPRE analysis from Python subprocess
# Usage: Rscript run_inspre_benchmark.R config.json

# Load required libraries
suppressPackageStartupMessages({
  library(hdf5r)
  library(dplyr)
  library(purrr)
  library(inspre)
  library(jsonlite)
})

config_file <- commandArgs(trailingOnly = TRUE)[1]
opt <- fromJSON(config_file)

# Main analysis function
run_inspre_analysis <- function() {

  tryCatch({
    hfile <- H5File$new(opt$input, "r")
    X_matrix <- hfile[['X']][,]
    var <- inspre::parse_hdf5_df(hfile, 'var')
  }, finally = {
    hfile$close()
  })

  # AnnData writes X as cells x genes, hdf5r reads it transposed as genes x cells.
  # fit_inspre_from_X expects samples x features (cells x genes), so we transpose.
  X <- t(X_matrix)
  colnames(X) <- var$gene_name
  targets_vector <- opt$targets

  cat("X dimensions (cells x genes):", nrow(X), "x", ncol(X), "\n")
  cat("Number of targets:", length(targets_vector), "\n")
  cat("Unique targets:", paste(unique(targets_vector), collapse=", "), "\n")

  cat("\n--- Data summary ---\n")
  cat("X class:", class(X), "\n")
  cat("X range: [", min(X, na.rm=TRUE), ",", max(X, na.rm=TRUE), "]\n")
  cat("X NA count:", sum(is.na(X)), "\n")
  cat("X Inf count:", sum(is.infinite(X)), "\n")
  cat("X zero fraction:", mean(X == 0, na.rm=TRUE), "\n")
  cat("Colnames (first 10):", paste(head(colnames(X), 10), collapse=", "), "\n")
  cat("Rownames (first 10):", paste(head(rownames(X), 10), collapse=", "), "\n")
  cat("Targets (first 10):", paste(head(targets_vector, 10), collapse=", "), "\n")
  cat("Targets matching colnames:", sum(unique(targets_vector) %in% colnames(X)), "of", length(unique(targets_vector[targets_vector != "non-targeting"])), "\n")
  cat("'non-targeting' count:", sum(targets_vector == "non-targeting"), "\n")
  cat("--------------------\n\n")

  start_time <- Sys.time()
  results <- fit_inspre_from_X(
    X = X,
    targets = targets_vector,
    weighted = opt$weighted,
    DAG = opt$dag,
    max_med_ratio = opt$max_med_ratio,
    nlambda = opt$nlambda,
    its = opt$iterations,
    verbose = opt$verbose,
    ncores = opt$ncores,
    cv_folds = opt$cv_folds
  )
  runtime <- as.numeric(difftime(Sys.time(), start_time, units="secs"))

  output_data <- list(
    runtime_seconds = runtime,
    n_genes = ncol(X),
    n_cells = nrow(X),
    n_targets = length(unique(targets_vector[targets_vector != "non-targeting"])),
    parameters = list(
      weighted = opt$weighted,
      dag = opt$dag,
      nlambda = opt$nlambda,
      iterations = opt$iterations,
      ncores = opt$ncores,
      max_med_ratio = opt$max_med_ratio
    )
  )

  cat("\n--- All fields returned by fit_inspre_from_X ---\n")
  cat("Names:", paste(names(results), collapse=", "), "\n")
  for (nm in names(results)) {
    val <- results[[nm]]
    if (is.matrix(val) || is.array(val)) {
      cat(sprintf("  %s: matrix/array, dim = %s\n", nm, paste(dim(val), collapse=" x ")))
    } else if (is.list(val)) {
      cat(sprintf("  %s: list, length = %d\n", nm, length(val)))
      for (nm2 in names(val)) {
        val2 <- val[[nm2]]
        if (is.matrix(val2) || is.array(val2)) {
          cat(sprintf("    %s: matrix/array, dim = %s\n", nm2, paste(dim(val2), collapse=" x ")))
        } else if (is.numeric(val2) && length(val2) == 1) {
          cat(sprintf("    %s: scalar = %g\n", nm2, val2))
        } else {
          cat(sprintf("    %s: %s, length = %d\n", nm2, class(val2), length(val2)))
        }
      }
    } else if (is.numeric(val) && length(val) == 1) {
      cat(sprintf("  %s: scalar = %g\n", nm, val))
    } else {
      cat(sprintf("  %s: %s, length = %d\n", nm, class(val), length(val)))
    }
  }
  cat("-------------------------------------------------\n\n")

  R_hat_mat <- as.matrix(results$R_hat)
  cat("\n--- Inferred network (R_hat) summary ---\n")
  cat("Dimensions:", dim(R_hat_mat), "\n")
  cat("Rownames (first 10):", paste(head(rownames(R_hat_mat), 10), collapse=", "), "\n")
  cat("Colnames (first 10):", paste(head(colnames(R_hat_mat), 10), collapse=", "), "\n")
  cat("Value range: [", min(R_hat_mat, na.rm=TRUE), ",", max(R_hat_mat, na.rm=TRUE), "]\n")
  cat("NA count:", sum(is.na(R_hat_mat)), "\n")
  nonzero_off_diag <- sum(R_hat_mat[row(R_hat_mat) != col(R_hat_mat)] != 0, na.rm=TRUE)
  total_off_diag <- nrow(R_hat_mat) * (nrow(R_hat_mat) - 1)
  cat("Non-zero off-diagonal entries:", nonzero_off_diag, "of", total_off_diag, "\n")
  cat("----------------------------------------\n\n")

  output_data$R_hat <- R_hat_mat
  output_data$R_hat_dimensions <- dim(results$R_hat)
  output_data$R_hat_rownames <- rownames(results$R_hat)
  output_data$R_hat_colnames <- colnames(results$R_hat)
  output_data$lambda_opt <- results$lambda_opt

  # Additional output fields for inspection from Python
  output_data$lambda <- as.numeric(results$lambda)
  output_data$rho <- as.numeric(results$rho)
  output_data$L <- as.numeric(results$L)
  output_data$train_error <- as.numeric(results$train_error)
  output_data$test_error <- as.numeric(results$test_error)
  output_data$gamma <- as.numeric(results$gamma)
  output_data$W <- as.matrix(results$W)
  output_data$W_rownames <- rownames(results$W)
  output_data$SE_hat <- as.matrix(results$SE_hat)
  # G_hat is D x D x nlambda — serialize as a list of nlambda matrices
  G_hat_arr <- results$G_hat
  n_lambda <- dim(G_hat_arr)[3]
  output_data$G_hat <- lapply(seq_len(n_lambda), function(i) as.matrix(G_hat_arr[,,i]))
  output_data$G_hat_dimensions <- dim(G_hat_arr)
  output_data$G_hat_rownames <- rownames(G_hat_arr[,,1])
  output_data$G_hat_colnames <- colnames(G_hat_arr[,,1])

  # Select best lambda by minimum CV eps_hat_G, or fall back to smallest lambda
  if (!is.null(results$eps_hat_G) && any(!is.nan(as.numeric(results$eps_hat_G)))) {
    eps_hat_G <- as.numeric(results$eps_hat_G)
    best_lambda_idx <- which.min(eps_hat_G)
    cat("\n--- Lambda selection by CV eps_hat_G ---\n")
    cat("Lambda values:", paste(round(results$lambda, 4), collapse=", "), "\n")
    cat("eps_hat_G:", paste(round(eps_hat_G, 4), collapse=", "), "\n")
    cat("Best lambda index:", best_lambda_idx, "\n")
    cat("Best lambda value:", results$lambda[best_lambda_idx], "\n")
    cat("-----------------------------------------\n\n")
  } else {
    best_lambda_idx <- length(results$lambda)
    cat("\n--- No CV performed, using smallest lambda (index", best_lambda_idx, ") ---\n\n")
  }

  # Diagnostic: print G_hat at best lambda before serialization
  G_hat_best <- G_hat_arr[,,best_lambda_idx]
  cat("\n--- G_hat at best lambda (R, index", best_lambda_idx, ") ---\n")
  cat("Rownames:", paste(rownames(G_hat_best), collapse=", "), "\n")
  cat("Colnames:", paste(colnames(G_hat_best), collapse=", "), "\n")
  print(G_hat_best)
  cat("---------------------------------------------------\n\n")

  output_data$best_lambda_idx <- best_lambda_idx
  output_data$best_lambda <- results$lambda[best_lambda_idx]

  write_json(output_data, opt$output, pretty=TRUE, auto_unbox=TRUE)
}

# Run the analysis
run_inspre_analysis()




