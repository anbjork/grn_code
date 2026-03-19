

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






def filter_data(nadata):
	"""
	### Data filtering
	# According to methods:2.10 in dspin bioarxiv
	# https://www.biorxiv.org/content/10.1101/2023.04.19.537364v4.full
	#
	# Deviation: 
	# They use their Poisson thing for highly variable genes.
	# I'll just use scanpys for now.
	# Possibly worth coming back to.
	#
	# Deviation:
	# They might use log transformed counts equivalent to
	# adata_log in the downstream analysis, but this uses the
	# the gem group z normalised data from Replogle, see nadata
	"""

	# from IPython import embed; embed(colors = 'Linux')

	radata = ad.read_h5ad('data/replogle/K562_gwps_raw_bulk_01.h5ad')
	adata_log = copy.deepcopy(radata)
	norm = 1e4 * radata.X / np.sum(radata.X, axis = 1)[:, None]
	adata_log.X = np.log(norm + 1)

	gene_names = [elem.split(sep = '_')[1] for elem in adata_log.obs.index]
	adata_log.obs.insert(loc = 0, column = 'gene_name', value = gene_names)



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

		highly_variable_genes = set(list(adata_tmp.var.gene_name[adata_tmp.var['highly_variable']]) + list(human_tfs['TF']))
		high_var_in_data = np.isin(
			element = adata_tmp.var.gene_name,
			test_elements = list(highly_variable_genes))


		bb = np.all(
			(
				include_gene_on_observation_based_conditions,
				high_var_in_data, 
				np.logical_not(genes_with_inf)
				),
			axis = 0
			)
		# Applying both observation and gene filters
		# Check that genes_with_inf are actually filtered out
		# In [38]: (bb & genes_with_inf).any()
		# Out[38]: False
		# Had a bug here before, using np.logical_and that can only
		# take 2 1D arrays. Now it looks good! ^^
		filtered_adatas[str(num_genes)] = nadata[
			combined_observation_filters,
			bb
			].copy()


		# from IPython import embed; embed(colors = 'Linux')


	# In [99]: filtered_adatas
	# Out[99]:
	# {'500': AnnData object with n_obs × n_vars = 3241 × 695
	#      obs: 'UMI_count_unfiltered', 'num_cells_unfiltered', 'num_cells_filtered', 'control_expr', 'fold_expr', 'pct_expr', 'core_control', 'mean_leverage_score', 'std_leverage_score', 'energy_test_p_value', 'anderson_darling_counts', 'mann_whitney_counts', 'z_gemgroup_UMI', 'mitopercent', 'TE_ratio', 'cnv_score_z'
	#      var: 'gene_name', 'mean', 'std', 'cv', 'in_matrix', 'gini', 'clean_mean', 'clean_std', 'clean_cv',
	#  '1000': AnnData object with n_obs × n_vars = 3241 × 885
	#      obs: 'UMI_count_unfiltered', 'num_cells_unfiltered', 'num_cells_filtered', 'control_expr', 'fold_expr', 'pct_expr', 'core_control', 'mean_leverage_score', 'std_leverage_score', 'energy_test_p_value', 'anderson_darling_counts', 'mann_whitney_counts', 'z_gemgroup_UMI', 'mitopercent', 'TE_ratio', 'cnv_score_z'
	#      var: 'gene_name', 'mean', 'std', 'cv', 'in_matrix', 'gini', 'clean_mean', 'clean_std', 'clean_cv'}

	# Some more genes and lots more conditions/perturbations than 
	# in the bioarxiv paper, but I'll proceed for now
	#
	# It's possible that they apply the observation based gene
	# conditions also to the observed genes

	# total_cells = {key: sum(aa.obs.num_cells_filtered) 
	# 	for key, aa in filtered_adatas.items()}
	# In [101]: total_cells
	# Out[101]: {'500': 602651.0, '1000': 602651.0}

	return(filtered_adatas)




# output_dir = Path('outputs')
# scanpy_retard_dir = Path('figures')
# [d.mkdir(exist_ok = True) for d in (output_dir, scanpy_retard_dir)]
#


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

wip = adata[:, np.isin(adata.var.gene_name, genes)]















# # # Got this from preprocess.py
# # def add_gene_names_to_observation_metadata(adata):	
# # 	gene_names = [elem.split(sep = '_')[1] for elem in adata.obs.index]
# # 	adata.obs.insert(loc = 0, column = 'gene_name', value = gene_names)
# # add_gene_names_to_observation_metadata(nadata)
#
#
# adata = filter_data(nadata)
#
#
# anton_util.pickle_object(adata, output_dir / 'replogle_with_dspin_filters.pkl')
#
#
#
# # # Now the new data
# # nadata = pd.read_csv(
# # 	data_paths['beeline_hESC'],
# # 	index_col = 0)
# # nadata.T
#
#
#
#
