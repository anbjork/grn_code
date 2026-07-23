#!/usr/bin/env Rscript

# Standalone R script for running INSPRE analysis from Python subprocess
# Usage: Rscript run_inspre_benchmark.R --input data.h5ad --output results.json [options]

# Load required libraries
suppressPackageStartupMessages({
  library(hdf5r)
  library(dplyr)
  library(inspre)
  library(jsonlite)
  library(optparse)
})

# Define command line options
option_list <- list(
  make_option(c("--input", "-i"), type="character", default=NULL,
              help="Input HDF5 file path", metavar="character"),
  make_option(c("--output", "-o"), type="character", default="inspre_results.json",
              help="Output JSON file path [default %default]", metavar="character"),
  make_option(c("--targets", "-t"), type="character", default=NULL,
              help="Comma-separated list of target genes", metavar="character"),
  make_option(c("--ncores", "-c"), type="integer", default=1,
              help="Number of CPU cores [default %default]", metavar="integer"),
  make_option(c("--weighted"), action="store_true", default=TRUE,
              help="Use weighted fitting [default %default]"),
  make_option(c("--dag"), action="store_true", default=FALSE,
              help="Enforce DAG constraints [default %default]"),
  make_option(c("--nlambda"), type="integer", default=20,
              help="Number of lambda values [default %default]", metavar="integer"),
  make_option(c("--iterations"), type="integer", default=100,
              help="Maximum iterations [default %default]", metavar="integer"),
  make_option(c("--verbose"), type="integer", default=1,
              help="Verbosity level (0-2) [default %default]", metavar="integer"),
  make_option(c("--filter"), action="store_true", default=TRUE,
              help="Filter low-quality perturbations [default %default]"),
  make_option(c("--max_med_ratio"), type="double", default=10.0,
              help="Maximum median weight ratio [default %default]", metavar="double")
)

# Parse command line arguments
opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

# Validate required arguments
if (is.null(opt$input)) {
  print_help(opt_parser)
  stop("Input file must be specified with --input", call.=FALSE)
}

if (!file.exists(opt$input)) {
  stop(paste("Input file does not exist:", opt$input), call.=FALSE)
}

# Function to safely close HDF5 file
safe_close_h5 <- function(hfile) {
  tryCatch({
    if (!is.null(hfile) && hfile$is_valid) {
      hfile$close()
    }
  }, error = function(e) {
    # Ignore errors when closing
  })
}

# Main analysis function
run_inspre_analysis <- function() {
  cat("Starting INSPRE analysis...\n")
  cat("Input file:", opt$input, "\n")
  cat("Output file:", opt$output, "\n")
  
  hfile <- NULL
  
  tryCatch({
    # Open HDF5 file
    cat("Opening HDF5 file...\n")
    hfile <- H5File$new(opt$input, "r")
    
    # Parse metadata
    cat("Parsing metadata...\n")
    obs_data <- parse_hdf5_df(hfile, 'obs')
    var_data <- parse_hdf5_df(hfile, 'var')
    
    cat("Data dimensions:\n")
    cat("- Cells:", nrow(obs_data), "\n")
    cat("- Genes:", nrow(var_data), "\n")
    
    # Get control cells
    cat("Identifying control cells...\n")
    cells_ntc <- obs_data$gene == "non-targeting"
    n_ntc <- sum(cells_ntc)
    cat("- Control cells:", n_ntc, "\n")
    
    if (n_ntc == 0) {
      stop("No non-targeting control cells found. Cells should have gene = 'non-targeting'")
    }
    
    # Extract control expression matrix
    X_control <- hfile[['X']][, cells_ntc]
    rownames(X_control) <- var_data$gene_id
    
    # Determine targets
    if (!is.null(opt$targets)) {
      # Use specified targets
      target_genes <- trimws(strsplit(opt$targets, ",")[[1]])
      cat("Using specified targets:", paste(target_genes, collapse=", "), "\n")
    } else {
      # Use all available targets
      target_genes <- unique(obs_data$gene[obs_data$gene != "non-targeting"])
      cat("Using all available targets (", length(target_genes), "):", 
          paste(head(target_genes, 5), collapse=", "), 
          if(length(target_genes) > 5) "..." else "", "\n")
    }
    
    # Create targets list
    targets_list <- as.list(target_genes)
    names(targets_list) <- target_genes
    
    # Validate targets exist in data
    available_targets <- unique(obs_data$gene[obs_data$gene != "non-targeting"])
    missing_targets <- setdiff(target_genes, available_targets)
    if (length(missing_targets) > 0) {
      warning("Some targets not found in data: ", paste(missing_targets, collapse=", "))
      targets_list <- targets_list[target_genes %in% available_targets]
    }
    
    if (length(targets_list) == 0) {
      stop("No valid targets found in data")
    }
    
    cat("Running INSPRE with", length(targets_list), "targets...\n")
    cat("Parameters:\n")
    cat("- Weighted:", opt$weighted, "\n")
    cat("- DAG constraints:", opt$dag, "\n")
    cat("- Lambda values:", opt$nlambda, "\n")
    cat("- Max iterations:", opt$iterations, "\n")
    cat("- CPU cores:", opt$ncores, "\n")
    
    # Run INSPRE analysis
    start_time <- Sys.time()
    
    results <- fit_inspre_from_h5X(
      X = hfile[['X']], 
      X_control = X_control,
      X_ids = obs_data$gene_transcript, 
      X_vars = var_data$gene_id, 
      targets = targets_list,
      weighted = opt$weighted,
      DAG = opt$dag,
      filter = opt$filter,
      max_med_ratio = opt$max_med_ratio,
      nlambda = opt$nlambda,
      its = opt$iterations,
      verbose = opt$verbose,
      ncores = opt$ncores
    )
    
    end_time <- Sys.time()
    runtime <- as.numeric(difftime(end_time, start_time, units="secs"))
    
    cat("Analysis completed in", round(runtime, 2), "seconds\n")
    
    # Prepare output
    output_data <- list(
      success = TRUE,
      runtime_seconds = runtime,
      n_genes = nrow(var_data),
      n_cells = nrow(obs_data),
      n_controls = n_ntc,
      n_targets = length(targets_list),
      targets = names(targets_list),
      parameters = list(
        weighted = opt$weighted,
        dag = opt$dag,
        nlambda = opt$nlambda,
        iterations = opt$iterations,
        ncores = opt$ncores,
        filter = opt$filter,
        max_med_ratio = opt$max_med_ratio
      )
    )
    
    # Add results
    if ("R_hat" %in% names(results)) {
      output_data$R_hat <- as.matrix(results$R_hat)
      output_data$R_hat_dimensions <- dim(results$R_hat)
      cat("Network matrix dimensions:", paste(dim(results$R_hat), collapse=" x "), "\n")
    }
    
    if ("lambda_opt" %in% names(results)) {
      output_data$lambda_opt <- results$lambda_opt
      cat("Optimal lambda:", results$lambda_opt, "\n")
    }
    
    # Add any other results
    other_fields <- setdiff(names(results), c("R_hat", "lambda_opt"))
    for (field in other_fields) {
      if (is.numeric(results[[field]]) || is.logical(results[[field]]) || is.character(results[[field]])) {
        output_data[[field]] <- results[[field]]
      }
    }
    
    # Save results
    cat("Saving results to:", opt$output, "\n")
    write_json(output_data, opt$output, pretty=TRUE, auto_unbox=TRUE)
    
    cat("INSPRE analysis completed successfully!\n")
    
  }, error = function(e) {
    cat("ERROR:", e$message, "\n")
    
    # Save error information
    error_data <- list(
      success = FALSE,
      error = e$message,
      parameters = list(
        input_file = opt$input,
        weighted = opt$weighted,
        dag = opt$dag,
        nlambda = opt$nlambda,
        iterations = opt$iterations,
        ncores = opt$ncores
      )
    )
    
    write_json(error_data, opt$output, pretty=TRUE, auto_unbox=TRUE)
    stop(e$message, call.=FALSE)
    
  }, finally = {
    # Always try to close the HDF5 file
    safe_close_h5(hfile)
  })
}

# Run the analysis
run_inspre_analysis()
