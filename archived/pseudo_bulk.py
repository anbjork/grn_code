import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path

data_set_name = 'K562_essential_raw_singlecell_01'
preprocessed_path = Path(f'data/replogle/{data_set_name}_preprocessed.h5ad')

wip = ad.read_h5ad(preprocessed_path)

# Pseudo bulk using scanpy
# Aggregate counts per perturbation group (gene column in obs)
pseudo_bulk = sc.get.aggregate(wip, by='gene', func='sum')

# sc.get.aggregate stores results in layers, move sum layer to X
pseudo_bulk.X = pseudo_bulk.layers['sum']

# Check cells per perturbation group
cells_per_gene = wip.obs.groupby('gene', observed=True).size()

print("=== Cells per perturbation group ===")
print(f"Total number of perturbation groups: {len(cells_per_gene)}")
print(f"\nSummary statistics:")
print(cells_per_gene.describe())

print(f"\nGroups with fewer than 10 cells:")
few_cells = cells_per_gene[cells_per_gene < 10]
print(few_cells if len(few_cells) > 0 else "None")

print(f"\nGroups with fewer than 30 cells:")
few_cells_30 = cells_per_gene[cells_per_gene < 30]
print(few_cells_30 if len(few_cells_30) > 0 else "None")

# Plot distribution of cells per group
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(cells_per_gene, bins=50)
axes[0].set_xlabel('Number of cells')
axes[0].set_ylabel('Number of perturbation groups')
axes[0].set_title('Distribution of cells per perturbation group')

axes[1].hist(cells_per_gene, bins=50, cumulative=True, density=True)
axes[1].set_xlabel('Number of cells')
axes[1].set_ylabel('Cumulative fraction of groups')
axes[1].set_title('Cumulative distribution of cells per perturbation group')

plt.tight_layout()
fig.savefig('figures/cells_per_perturbation_group.png')
plt.close()
print("\nSaved figure to figures/cells_per_perturbation_group.png")

# Filter out perturbation groups with fewer than 30 cells
min_cells = 30
groups_to_keep = cells_per_gene[cells_per_gene >= min_cells].index
n_removed = len(cells_per_gene) - len(groups_to_keep)
print(f"\nFiltering out {n_removed} perturbation groups with fewer than {min_cells} cells.")
pseudo_bulk = pseudo_bulk[np.isin(pseudo_bulk.obs_names, groups_to_keep)].copy()
print(f"Remaining perturbation groups: {pseudo_bulk.shape[0]}")

# Check the pseudo bulk result
print("\n=== Pseudo bulk result ===")
print(f"Shape: {pseudo_bulk.shape}  (perturbation groups x genes)")
print(f"\nFirst few rows of pseudo bulk obs metadata:")
print(pseudo_bulk.obs.head())

# Write pseudo bulk to file
pseudo_bulk_path = Path(f'data/replogle/{data_set_name}_pseudo_bulk.h5ad')
print(f"\nSaving pseudo bulk to {pseudo_bulk_path} ...")
pseudo_bulk.write_h5ad(pseudo_bulk_path)
print("Saved.")
