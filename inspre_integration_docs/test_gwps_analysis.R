#!/usr/bin/env Rscript

# Test version of the GWPS analysis to check functionality
# This version removes the hardcoded paths and focuses on testing the interface

# Load required libraries
cat("Loading libraries...\n")
library(hdf5r, lib.loc = "/home/anbjork/R/x86_64-pc-linux-gnu-library/4.3")
library(dplyr)
library(purrr) 
library(inspre, lib.loc = "/home/anbjork/R/x86_64-pc-linux-gnu-library/4.3")

cat("Libraries loaded successfully!\n")

# Check what functions are available in inspre
cat("\n=== Available inspre functions ===\n")
inspre_functions <- ls("package:inspre")
print(inspre_functions)

# Look specifically for HDF5-related functions
cat("\n=== HDF5-related functions ===\n")
h5_functions <- inspre_functions[grepl("h5|H5", inspre_functions)]
if(length(h5_functions) > 0) {
  print(h5_functions)
} else {
  cat("No obvious HDF5 functions found, checking for parse and calc functions...\n")
  parse_calc_functions <- inspre_functions[grepl("parse|calc", inspre_functions)]
  print(parse_calc_functions)
}

# Check if the key functions exist
cat("\n=== Checking key functions ===\n")
key_functions <- c("parse_hdf5_df", "calc_inst_effect_h5X", "fit_inspre_from_h5X")
for(func in key_functions) {
  if(exists(func, where = "package:inspre")) {
    cat("✓", func, "exists\n")
  } else {
    cat("✗", func, "NOT FOUND\n")
  }
}

# Try to get help for the main HDF5 function
cat("\n=== Function documentation ===\n")
tryCatch({
  help("fit_inspre_from_h5X", package = "inspre")
  cat("Help for fit_inspre_from_h5X available\n")
}, error = function(e) {
  cat("No help available for fit_inspre_from_h5X:", e$message, "\n")
})

# Check what the original paths were supposed to contain
cat("\n=== Original file paths (for reference) ===\n")
original_paths <- c(
  '/gpfs/commons/groups/knowles_lab/gwps/data/K562_essential_normalized_singlecell_01.h5ad',
  '/gpfs/commons/groups/knowles_lab/gwps/saved_rdata/gwps_res_dag.Rdata',
  '/gpfs/commons/groups/knowles_lab/gwps/saved_rdata/gwps_res_nodag.Rdata',
  '/gpfs/commons/groups/knowles_lab/gwps/saved_rdata/gwps_guide_data.Rdata'
)

for(path in original_paths) {
  if(file.exists(path)) {
    cat("Found:", path, "\n")
  } else {
    cat("Missing:", path, "\n")
  }
}

cat("\n=== Creating synthetic test data ===\n")
# Create a small synthetic HDF5 file to test the interface
test_h5_file <- "test_data.h5ad"

if (!file.exists(test_h5_file)) {
  cat("Creating synthetic HDF5 test file...\n")
  
  # Create synthetic data
  n_genes <- 50
  n_cells_per_target <- 20
  n_targets <- 10
  n_ntc <- 100
  
  # Gene names
  gene_ids <- paste0("GENE", 1:n_genes)
  
  # Create targets (subset of genes that will be perturbed)
  target_genes <- gene_ids[1:n_targets]
  
  # Create cell metadata (obs)
  obs_data <- data.frame(
    gene_transcript = c(
      rep("non-targeting", n_ntc),
      rep(target_genes, each = n_cells_per_target)
    ),
    gene = c(
      rep("non-targeting", n_ntc),
      rep(target_genes, each = n_cells_per_target)
    ),
    gene_id = c(
      rep("non-targeting", n_ntc),
      rep(target_genes, each = n_cells_per_target)
    ),
    stringsAsFactors = FALSE
  )
  
  # Create gene metadata (var)
  var_data <- data.frame(
    gene_id = gene_ids,
    cv = runif(n_genes, 0.1, 2.0),  # coefficient of variation
    stringsAsFactors = FALSE
  )
  
  # Create expression matrix (genes x cells)
  n_total_cells <- nrow(obs_data)
  set.seed(42)
  X_data <- matrix(rnorm(n_genes * n_total_cells, mean = 5, sd = 2), 
                   nrow = n_genes, ncol = n_total_cells)
  
  # Add some perturbation effects (reduce expression of target genes in their respective cells)
  for (i in 1:n_targets) {
    target_idx <- which(gene_ids == target_genes[i])
    cell_start <- n_ntc + (i-1) * n_cells_per_target + 1
    cell_end <- n_ntc + i * n_cells_per_target
    # Reduce target gene expression in its perturbed cells
    X_data[target_idx, cell_start:cell_end] <- X_data[target_idx, cell_start:cell_end] - 2
  }
  
  # Create HDF5 file
  h5file <- H5File$new(test_h5_file, "w")
  
  # Write data matrix
  h5file[["X"]] <- X_data
  
  # Write obs metadata
  obs_group <- h5file$create_group("obs")
  for (col in names(obs_data)) {
    obs_group[[col]] <- obs_data[[col]]
  }
  
  # Write var metadata  
  var_group <- h5file$create_group("var")
  for (col in names(var_data)) {
    var_group[[col]] <- var_data[[col]]
  }
  
  h5file$close()
  cat("Created test HDF5 file:", test_h5_file, "\n")
  cat("Data dimensions:", n_genes, "genes x", n_total_cells, "cells\n")
  cat("Targets:", paste(target_genes, collapse = ", "), "\n")
}

cat("\n=== Testing HDF5 interface ===\n")
tryCatch({
  # Test reading the HDF5 file
  hfile_test <- H5File$new(test_h5_file, "r")
  
  # Parse metadata
  obs_test <- parse_hdf5_df(hfile_test, 'obs')
  var_test <- parse_hdf5_df(hfile_test, 'var')
  
  cat("Successfully parsed HDF5 metadata:\n")
  cat("- obs dimensions:", nrow(obs_test), "x", ncol(obs_test), "\n")
  cat("- var dimensions:", nrow(var_test), "x", ncol(var_test), "\n")
  cat("- Unique targets:", length(unique(obs_test$gene[obs_test$gene != "non-targeting"])), "\n")
  
  # Test a simple analysis with DAG=FALSE
  cat("\n=== Testing inspre with DAG=FALSE ===\n")
  
  # Get non-targeting controls
  cells_ntc <- obs_test$gene == "non-targeting"
  n_ntc <- sum(cells_ntc)
  X_ntc <- hfile_test[['X']][, cells_ntc]
  rownames(X_ntc) <- var_test$gene_id
  
  # Get a few targets to test
  test_targets <- unique(obs_test$gene[obs_test$gene != "non-targeting"])[1:3]
  targets_list <- as.list(test_targets)
  names(targets_list) <- test_targets
  
  cat("Testing with", length(targets_list), "targets:", paste(test_targets, collapse = ", "), "\n")
  
  # Run a quick test with minimal parameters
  res_test <- fit_inspre_from_h5X(
    hfile_test[['X']], X_ntc, obs_test$gene_transcript, var_test$gene_id, targets_list,
    weighted = TRUE,   # Enable weighting for better results
    DAG = FALSE,       # Use DAG=FALSE as recommended
    nlambda = 3,       # Just a few lambda values for speed
    its = 10,          # Fewer iterations for speed
    verbose = 2,       # Show detailed output
    ncores = 1         # Single core for simplicity
  )
  
  cat("SUCCESS! inspre ran without errors\n")
  cat("Result structure:\n")
  print(names(res_test))
  
  if ("R_hat" %in% names(res_test)) {
    cat("R_hat dimensions:", dim(res_test$R_hat), "\n")
  }
  
  hfile_test$close()
  
}, error = function(e) {
  cat("Error testing HDF5 interface:", e$message, "\n")
  if (exists("hfile_test")) {
    try(hfile_test$close(), silent = TRUE)
  }
})

cat("\n=== Test completed ===\n")
cat("If successful, the HDF5 interface is working and ready for your benchmark data!\n")
