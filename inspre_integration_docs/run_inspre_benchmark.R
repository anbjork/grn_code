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
    ncores = opt$ncores
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

  write_json(output_data, opt$output, pretty=TRUE, auto_unbox=TRUE)
}

# Run the analysis
run_inspre_analysis()




