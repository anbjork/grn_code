import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import scanpy as sc
import genesnake as gs

import pickle
from pathlib import Path
import copy
import traceback

import anton_util


essential_raw_single_cell_dataset = '/home/anbjork/projects/replogle_round_2/data/replogle/K562_essential_raw_singlecell_01.h5ad'
fp = Path(essential_raw_single_cell_dataset)

adata = ad.read_h5ad(fp)




gene_excel = 'tmp/41467_2025_64353_MOESM4_ESM.xlsx'
excel = pd.read_excel(gene_excel, sheet_name = None)

# The top most occurring genes in the dataset
from collections import Counter
df = pd.DataFrame(zip(*Counter(Counter(adata.obs.gene).values()).items())).T
df.sort_values(by = 1)
df.sort_values(by = 0)




gene_excel = 'tmp/41467_2025_64353_MOESM4_ESM.xlsx'
excel = pd.read_excel(gene_excel, sheet_name = None, header = 1)
dfg = excel['Supplementary Data 2']
genes = dfg['gene_name']

wip = adata[:, np.isin(adata.var.gene_name, genes)].copy()

# Calculate QC metrics
sc.pp.calculate_qc_metrics(wip, inplace=True)

# Visualise QC metrics to assess cell health
# Using upstream mitopercent as MT genes are not in our gene subset
#
# Based on results, it looks like they've filtered some already
sc.pl.violin(wip, ['n_genes_by_counts', 'total_counts', 'mitopercent'],
    jitter=0.4, multi_panel=True, save='_qc_violin.png')
sc.pl.scatter(wip, x='total_counts', y='n_genes_by_counts',
    save='_counts_vs_genes.png')
sc.pl.scatter(wip, x='total_counts', y='mitopercent',
    save='_counts_vs_mt.png')

# Cell filtering
sc.pp.filter_cells(wip, min_genes=200)      # Remove cells with too few genes detected
sc.pp.filter_cells(wip, min_counts=500)     # Remove cells with too few total counts

wip = wip[wip.obs.mitopercent < 20].copy()

# Save checkpoint before doublet detection, which is memory intensive
checkpoint_path = Path('tmp/wip_pre_scrublet.h5ad')
print(f'Saving checkpoint to {checkpoint_path} ...')
wip.write_h5ad(checkpoint_path)
print('Checkpoint saved.')

# Skipping this for now. Might have already been done by the original authors, and it's memory intensive. Can revisit if needed.
# # Doublet detection using Scrublet
# # Adds 'doublet_score' and 'predicted_doublet' columns to wip.obs
# # Using reduced sim_doublet_ratio=0.5 (default 2.0) to reduce memory usage
# # given ~310k cells
# sc.pp.scrublet(wip, sim_doublet_ratio=0.5)
# sc.pl.scrublet_score_distribution(wip, save='_scrublet.png')
# wip = wip[~wip.obs.predicted_doublet].copy()
