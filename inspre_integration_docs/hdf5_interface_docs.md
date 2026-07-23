# INSPRE HDF5 Interface Documentation

## Overview
The INSPRE package provides an HDF5 interface for analyzing perturbation data (e.g., Perturb-seq, CRISPR screens). This document explains the expected data format and how to use the interface.

## HDF5 File Structure

The HDF5 file should contain the following groups and datasets:

### Required Structure
```
file.h5ad
├── X                    # Main data matrix (genes × cells)
├── obs/                 # Cell metadata group
│   ├── gene            # Target gene for each cell
│   ├── gene_transcript # Target gene transcript (can be same as gene)
│   └── gene_id         # Target gene ID (can be same as gene)
└── var/                 # Gene metadata group
    ├── gene_id         # Gene identifiers
    └── cv              # Coefficient of variation (optional)
```

### Data Details

#### X Matrix
- **Dimensions**: genes × cells (genes as rows, cells as columns)
- **Type**: Numeric expression values (log-normalized recommended)
- **Missing values**: Should be handled before creating HDF5 file

#### obs (Cell Metadata)
- **gene**: Target gene name for each cell. Use "non-targeting" for control cells
- **gene_transcript**: Target transcript (often same as gene)
- **gene_id**: Target gene identifier (often same as gene)
- **Length**: Must match number of columns in X matrix

#### var (Gene Metadata)  
- **gene_id**: Gene identifiers, must match row names conceptually
- **cv**: Coefficient of variation (optional, used for filtering)
- **Length**: Must match number of rows in X matrix

## Control Cells
- Non-targeting control cells should have gene = "non-targeting"
- These are used as the baseline for calculating perturbation effects
- Recommend at least 50-100 control cells for stable results

## Target Specification
The `targets` parameter should be a named list where:
- Names are target identifiers
- Values are the target gene names (matching the "gene" field in obs)

Example:
```r
targets_list <- list(
  "GENE1" = "GENE1",
  "GENE2" = "GENE2", 
  "GENE3" = "GENE3"
)
```

## Key Parameters for fit_inspre_from_h5X()

### Essential Parameters
- **X**: HDF5 dataset containing the expression matrix
- **X_control**: Control expression matrix (non-targeting cells)
- **X_ids**: Vector of target gene names for each cell
- **X_vars**: Vector of gene identifiers
- **targets**: Named list of targets to analyze

### Recommended Settings
- **weighted = TRUE**: Use weighted fitting (recommended for better results)
- **DAG = FALSE**: Don't enforce DAG constraints (better accuracy, slower)
- **filter = TRUE**: Filter low-quality perturbations
- **max_med_ratio = 10**: Limit maximum weight ratios
- **verbose = 2**: Show detailed progress

### Performance Parameters
- **ncores**: Number of CPU cores to use
- **nlambda = 20**: Number of regularization parameters to test
- **its = 100**: Maximum iterations for convergence

## Example Usage

```r
library(hdf5r)
library(inspre)

# Open HDF5 file
hfile <- H5File$new("data.h5ad", "r")

# Parse metadata
obs_data <- parse_hdf5_df(hfile, 'obs')
var_data <- parse_hdf5_df(hfile, 'var')

# Get control cells
cells_ntc <- obs_data$gene == "non-targeting"
X_control <- hfile[['X']][, cells_ntc]
rownames(X_control) <- var_data$gene_id

# Define targets
targets_list <- list(
  "GENE1" = "GENE1",
  "GENE2" = "GENE2"
)

# Run INSPRE
results <- fit_inspre_from_h5X(
  X = hfile[['X']], 
  X_control = X_control,
  X_ids = obs_data$gene_transcript, 
  X_vars = var_data$gene_id, 
  targets = targets_list,
  weighted = TRUE,
  DAG = FALSE,
  verbose = 2,
  ncores = 4
)

hfile$close()
```

## Output Structure
The function returns a list containing:
- **R_hat**: Estimated regulatory network matrix
- **lambda_opt**: Optimal regularization parameter
- **convergence**: Convergence information
- Additional diagnostic information

## Python Integration Notes

### Creating HDF5 from Python
```python
import h5py
import numpy as np
import pandas as pd

# Create HDF5 file
with h5py.File('data.h5ad', 'w') as f:
    # Write expression matrix (genes × cells)
    f['X'] = expression_matrix
    
    # Write cell metadata
    obs_grp = f.create_group('obs')
    obs_grp['gene'] = cell_targets.astype('S')  # String data
    obs_grp['gene_transcript'] = cell_targets.astype('S')
    obs_grp['gene_id'] = cell_targets.astype('S')
    
    # Write gene metadata
    var_grp = f.create_group('var')
    var_grp['gene_id'] = gene_names.astype('S')
    var_grp['cv'] = gene_cv_values
```

### Calling from Python subprocess
```python
import subprocess
import json

# Prepare arguments
args = {
    'input_file': 'data.h5ad',
    'output_file': 'results.json',
    'targets': ['GENE1', 'GENE2', 'GENE3'],
    'ncores': 4
}

# Call R script
result = subprocess.run([
    'Rscript', 'run_inspre_benchmark.R',
    '--input', args['input_file'],
    '--output', args['output_file'], 
    '--targets', ','.join(args['targets']),
    '--ncores', str(args['ncores'])
], capture_output=True, text=True)
```

## Troubleshooting

### Common Issues
1. **Dimension mismatch**: Ensure X matrix dimensions match obs/var lengths
2. **Missing controls**: Must have cells with gene = "non-targeting"
3. **String encoding**: Use proper string encoding for HDF5 (UTF-8 or ASCII)
4. **Memory issues**: For large datasets, consider chunking or reducing nlambda

### Performance Tips
1. Use multiple cores (ncores parameter)
2. Filter genes with low variance before analysis
3. Start with fewer targets for testing
4. Use DAG=TRUE for faster convergence on difficult datasets
