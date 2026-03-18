
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


# # Old version. REMOVE when fixed
# #
# # Messy import since GeneSnake package is not packaged correctly.
# # According to docs
# # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
# import importlib.util
# import sys
# def import_from_path(module_name, file_path):
# 	spec = importlib.util.spec_from_file_location(module_name, file_path)
# 	module = importlib.util.module_from_spec(spec)
# 	sys.modules[module_name] = module
# 	spec.loader.exec_module(module)
# 	return module
# s = '../../genesnake_install/genesnake/methods/grn_inference.py'
# gsi = import_from_path('grn_inference', file_path = s)



output_dir = Path('outputs')
scanpy_retard_dir = Path('figures')
[d.mkdir(exist_ok = True) for d in (output_dir, scanpy_retard_dir)]


def add_gene_names_to_observation_metadata(adata):	
	gene_names = [elem.split(sep = '_')[1] for elem in adata.obs.index]
	adata.obs.insert(loc = 0, column = 'gene_name', value = gene_names)



def filter_data(nadata):

	radata = ad.read_h5ad('data/replogle/K562_gwps_raw_bulk_01.h5ad')
	adata_log = copy.deepcopy(radata)
	norm = 1e4 * radata.X / np.sum(radata.X, axis = 1)[:, None]
	adata_log.X = np.log(norm + 1)

	[add_gene_names_to_observation_metadata(adata) 
		for adata in [adata_log, nadata]]

	### Observation filtering 
	# Adapted from D-SPIN bioarxiv methods section 2.1, 
	# start at pdf page 66 of
	# https://www.biorxiv.org/content/10.1101/2023.04.19.537364v4.full.pdf
	# 
	# Note, it's not clearly stated in methods 2.10 that they apply
	# this observation filtering also for the benchmarking, but assuming
	# this is general to all steps for now. Otherwise, there is
	# almost no observation filtering which doesn't make sense.
	enough_cells = nadata.obs.num_cells_filtered >= 20
	controls = np.logical_and(
		nadata.obs['core_control'], 
		nadata.obs['num_cells_filtered'] >= 200)
	many_degs = nadata.obs.anderson_darling_counts >= 10
	# Numbers very similar to D-SPIN, but not identical.
	# Good enough for now
	# In [78]: controls.sum()
	# Out[78]: 105
	# # 105 in the D-SPIN bioarxiv
	# In [79]: many_degs.sum()
	# Out[79]: 3162
	# In [80]: many_degs.sum() / nadata.shape[0]
	# Out[80]: 0.2808669390655534

	# Control cells with non-targeting sgRNAs to include based on
	# methods section 2.1
	controls_from_2_10 = np.logical_and(
		adata_log.obs.gene_name == 'non-targeting', 
		adata_log.obs.num_cells_filtered > 250)
	# In [72]: controls_from_2_10.sum()
	# Out[72]: 44
	# dspin bioarxiv says 46, so that's encouragingly close

	combined_observation_filters = np.logical_and(
		enough_cells,
		# Need to use np.any, since >2 input arrays
		# np.logical_or can only take 2 1D arrays
		np.any(
			(
				controls,
				many_degs,
				controls_from_2_10
				), 
			axis = 0)
		)

	### Genes filtering
	# Observation based inclusion conditions for genes
	# Conditions according to 2.10 of dspin bioarxiv
	# Seems like dspin has a slightly different dataset than me.
	# Names of metadata fields are slighly different.
	# Or they have just found an explainer with longer names somewhere
	conditions = [
		adata_log.obs.num_cells_filtered > 20,
		adata_log.obs.anderson_darling_counts > 2,
		adata_log.obs.pct_expr < -0.05
		]
	genes_passing = adata_log.obs.gene_name[np.logical_and(*conditions)]
	include_gene_on_observation_based_conditions = np.isin(
		element = adata_log.var.gene_name, test_elements = genes_passing)

	### Infinite values in data
	# More comments on this in earlier versions, like 3
	genes_with_inf = np.any(np.isinf(nadata.X), axis = 0)


	# # Filter TFs to those expressed in > 0.05 of cells
	# human_tfs = pd.read_csv('data/beeline/networks/human-tfs.csv')['TF']
	# tmp = human_tfs[np.isin(human_tfs, nadata.obs.gene_name)]
	# tf_cell_fractions = {}
	# counts = []
	# for tf in tmp:
	# 	subset = nadata.obs.loc[
	# 		nadata.obs.gene_name == tf,
	# 		'num_cells_filtered'
	# 		]
	# 	tf_cell_fractions[tf] = subset.sum()
	# 	counts.append(subset.shape[0])
	# total_cells = nadata.obs.num_cells_filtered.sum()
	# tf_cell_fractions = pd.Series(tf_cell_fractions) / total_cells
	# # In [39]: max(tf_cell_fractions)
	# # Out[39]: 0.00048754057393075314
	# # Soo, no tfs expressed in > 5% of cells.
	# # dspin says they include a subset of those expressed in > 5% 
	# # of cells. Something is fishy..
	# # 
	# # Unfiltered cell counts?
	# tmp = human_tfs[np.isin(human_tfs, nadata.obs.gene_name)]
	# tf_cell_fractions = {}
	# counts = []
	# for tf in tmp:
	# 	subset = nadata.obs.loc[
	# 		nadata.obs.gene_name == tf,
	# 		'num_cells_unfiltered'
	# 		]
	# 	tf_cell_fractions[tf] = subset.sum()
	# 	counts.append(subset.shape[0])
	# total_cells = nadata.obs.num_cells_filtered.sum()
	# tf_cell_fractions = pd.Series(tf_cell_fractions) / total_cells
	# # In [92]: max(tf_cell_fractions)
	# # Out[92]: 0.0004991008143435442
	# # No..
	# # 
	# # Do they mean ones which have nonzero expression in
	# # > 5% of the observations/perturbation conditions?
	# tmp = human_tfs[np.isin(human_tfs, nadata.var.gene_name)]
	# nonzero_expressions = {}
	# for tf in tmp:
	# 	subset = nadata[:, nadata.var.gene_name == tf]
	# 	nonzero_expressions[tf] = subset.X.nonzero()[0].shape[0]
	# tf_expr_fractions = pd.Series(nonzero_expressions) / nadata.shape[0]
	# # Also don't sense, since that wouldn't filter any out
	# # In [89]: min(tf_expr_fractions)
	# # Out[89]: 0.9910286018831054
	# #
	# # Hmm, the pseudo bulk data cannot say anything about
	# # the number of cells where expression is non-zero, 
	# # because pseudobulk data has no statistics about indiviudal cells,
	# # they have statistics about each pseudo bulk
	# #
	# # Downloading the actual single cell data. It's 60GB, so will
	# # have to offload from memory to disk or such.


	# Get TFs from Beeline
	human_tfs = pd.read_csv('data/beeline/networks/human-tfs.csv')

	filtered_adatas = {}
	for num_genes in [500, 1000]:

		adata_tmp = adata_log.copy()
		sc.pp.highly_variable_genes(adata_tmp, n_top_genes = num_genes)
		sc.pl.highly_variable_genes(adata_tmp, save = '.svg')
		# Scanpy doesn't allow specification of file name and location -.-
		(scanpy_retard_dir / 'filter_genes_dispersion.svg').rename(
			output_dir / 
			f'cell_normalised_highly_variable_genes_top_{num_genes}.svg')

		tmp = adata_tmp.var.gene_name[adata_tmp.var['highly_variable']]
		highly_variable_genes = pd.Series(
			list(set(list(tmp) + list(human_tfs['TF']))))
		high_var_in_data = np.isin(
			element = adata_tmp.var.gene_name,
			test_elements = highly_variable_genes)

		bb = np.all(
			(
				include_gene_on_observation_based_conditions,
				high_var_in_data, 
				np.logical_not(genes_with_inf)
				),
			axis = 0
			)

		# Get observations where perturbed gene is in variables
		# non-targeting is the gene name for control cells
		genes_kept = list(nadata.var.gene_name[bb]) + ['non-targeting']
		perturbation_of_kept_gene = np.isin(nadata.obs.gene_name, genes_kept)

		# Applying both observation and gene filters
		# Check that genes_with_inf are actually filtered out
		# In [38]: (bb & genes_with_inf).any()
		# Out[38]: False
		# Had a bug here before, using np.logical_and that can only
		# take 2 1D arrays. Now it looks good! ^^
		fadata = nadata[
			np.logical_and(
				# Try filtering perturbations on genes in variables.
				# The dspin preprint never explicitly says they do
				# this, but dataset dimensions match much close
				# if one does, and it might make conceptual sense,
				# so testing it.
				perturbation_of_kept_gene,
				combined_observation_filters
				),
			bb
			].copy()

		filtered_adatas[str(num_genes)] = fadata

		# import ipdb; ipdb.set_trace()

	return(filtered_adatas)





data_paths = {
	'replogle': 'data/replogle/K562_gwps_normalized_bulk_01.h5ad',
	# 'beeline_hESC': 'data/beeline/hESC/ExpressionData.csv'
	}

radata = ad.read_h5ad('data/replogle/K562_gwps_raw_bulk_01.h5ad')
nadata = ad.read_h5ad(data_paths['replogle'])

adata_log = copy.deepcopy(radata)
norm = 1e4 * radata.X / np.sum(radata.X, axis = 1)[:, None]
adata_log.X = np.log(norm + 1)

filtered = {}
d = {
    "log-norm": adata_log,
    "z-norm": nadata
	}
for dataset_name, dataset in d.items():
	adata = filter_data(dataset)
	anton_util.pickle_object(
		adata, 
		output_dir / f'replogle_{dataset_name}_dspin_filters.pkl')
	filtered[dataset_name] = adata


# Observation filtering is very different:
# In [9]: filtered['log-norm']['500'].obs['num_cells_filtered'].sum()
# Out[9]: np.float64(602651.0)
# In [10]: filtered['log-norm']['500']
# Out[10]:
# AnnData object with n_obs × n_vars = 3241 × 695
#
# dspin bioarxiv
# After filtering, the TFs+500 dataset comprised 103k cells, 421 genes, and 467 conditions, while the TFs+1000 dataset included 150k cells, 624 genes, and 670 conditions.
# 
# Testing only including observations with perturbed gene in variables.
# Shapes are much closer. But not sure if that is what they did.



