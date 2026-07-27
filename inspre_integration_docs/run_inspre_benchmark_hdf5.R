#!/usr/bin/env Rscript

# Standalone R script for running INSPRE analysis from Python subprocess
# using the fit_inspre_from_h5X interface, mirroring the gwps analysis.
# Usage: Rscript run_inspre_benchmark_hdf5.R config.json

suppressPackageStartupMessages({
  library(hdf5r)
  library(dplyr)
  library(purrr)
  library(inspre)
  library(jsonlite)
})

config_file <- commandArgs(trailingOnly = TRUE)[1]
opt <- fromJSON(config_file)

run_inspre_analysis <- function() {

  # Load the hdf5 file. The X matrix is stored genes x cells (as written by Python).
  hfile <- H5File$new(opt$input, "r")

  var <- inspre::parse_hdf5_df(hfile, 'var')
  obs <- inspre::parse_hdf5_df(hfile, 'obs')

  cat("obs columns:", paste(colnames(obs), collapse=", "), "\n")
  cat("var columns:", paste(colnames(var), collapse=", "), "\n")
  cat("X dimensions (genes x cells):", hfile[['X']]$dims[1], "x", hfile[['X']]$dims[2], "\n")
  cat("Number of obs rows:", nrow(obs), "\n")
  cat("Unique perturbations:", paste(unique(obs$perturbation), collapse=", "), "\n")

  ntc <- 'non-targeting'

  # Load NTC cells into memory (X is genes x cells).
  cat("Finding and loading NTC cells.\n")
  cells_ntc <- obs$perturbation == ntc
  n_ntc <- sum(cells_ntc)
  cat("NTC cell count:", n_ntc, "\n")

  X_ntc <- hfile[['X']][, cells_ntc]
  rownames(X_ntc) <- var$gene_name

  # Build targets list: named by guide id (obs$guide_id), values are gene names.
  # Only keep genes that appear in var (i.e. have expression data).
  genes_guides <- filter(obs, perturbation != ntc) %>%
    distinct(perturbation, guide_id) %>%
    filter(perturbation %in% var$gene_name)

  cat("Number of gene/guide pairs:", nrow(genes_guides), "\n")

  # Calculate guide effects to select effective guides, mirroring gwps analysis.
  cat("Calculating guide effect sizes.\n")
  guide_effects <- map2(
    genes_guides$guide_id,
    genes_guides$perturbation,
    ~ calc_inst_effect_h5X(.x, .y, hfile[['X']], X_ntc, obs$guide_id, var$gene_name)
  ) %>% list_rbind()

  guide_effects <- guide_effects %>%
    mutate(
      Z = inst_cor / cor_se,
      p = pt(Z, df = n - 2),
      p_adj = p.adjust(p, method = 'fdr')
    ) %>%
    arrange(target, p_adj) %>%
    filter(!duplicated(target))

  cat("Guide effects summary:\n")
  print(summary(guide_effects$inst_beta))

  # Filter: keep guides with inst_beta < -0.75 and at least 50 targeting cells.
  keep_guides <- filter(guide_effects, inst_beta < -0.75, n > n_ntc + 50)
  cat("Keeping", nrow(keep_guides), "genes/guides after filtering.\n")

  # Reorder to match the order variables appear in hfile.
  keep_guides <- keep_guides[
    match(
      var$gene_name[var$gene_name %in% keep_guides$target],
      keep_guides$target
    ), ]

  targets <- as.list(keep_guides$target)
  names(targets) <- keep_guides$inst_id

  cat("Final targets count:", length(targets), "\n")
  cat("Target genes (first 10):", paste(head(unlist(targets), 10), collapse=", "), "\n")

  start_time <- Sys.time()
  cat("Fitting inspre with fit_inspre_from_h5X.\n")
  res <- fit_inspre_from_h5X(
    hfile[['X']],
    X_ntc,
    obs$guide_id,
    var$gene_name,
    targets,
    max_med_ratio = opt$max_med_ratio,
    cv_folds     = 0,
    ncores       = opt$ncores,
    DAG          = opt$dag,
    min_nz       = 0.03 / 2,
    lambda_min_ratio = 0.1,
    nlambda      = opt$nlambda,
    verbose      = opt$verbose
  )
  hfile$close()
  runtime <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  cat("Fitting time (s):", runtime, "\n")

  R_hat_mat <- as.matrix(res$R_hat)
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

  output_data <- list(
    runtime_seconds  = runtime,
    n_genes          = nrow(R_hat_mat),
    parameters       = list(
      dag            = opt$dag,
      nlambda        = opt$nlambda,
      ncores         = opt$ncores,
      max_med_ratio  = opt$max_med_ratio
    ),
    R_hat            = R_hat_mat,
    R_hat_dimensions = dim(R_hat_mat),
    R_hat_rownames   = rownames(R_hat_mat),
    R_hat_colnames   = colnames(R_hat_mat)
  )

  write_json(output_data, opt$output, pretty=TRUE, auto_unbox=TRUE)
}

run_inspre_analysis()
