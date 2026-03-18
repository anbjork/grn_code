
import numpy as np
import pandas as pd
import anton_util
from pathlib import Path
import genesnake as gs

output_dir = Path('outputs')
[d.mkdir(exist_ok = True) for d in (output_dir, )]

def get_Y_and_P(
		adata, 
		gene_identifier_column_name,
		knockdown_value = -1):
	Y = pd.DataFrame(adata.X)
	Y.index = adata.obs[gene_identifier_column_name]
	Y.index.name = 'perturbations'
	Y.columns = adata.var[gene_identifier_column_name]
	Y.columns.name = 'genes'

	rows = np.array(Y.index)
	cols = np.array(Y.columns)
	M, N = Y.shape
	P_np = np.zeros(Y.shape)
	for ii in range(M):
		for jj in range(N):		
			if rows[ii] == cols[jj]:
				P_np[ii, jj] = knockdown_value

	P = pd.DataFrame(P_np)
	P.index = rows
	P.index.name = 'perturbations'
	P.columns = cols
	P.columns.name = 'genes'
	
	d = {'Y': Y, 'P': P}
	return(d)



all_adata = anton_util.unpickle_object(
	output_dir / 'replogle_log-norm_dspin_filters.pkl')
adata = all_adata['500']


ynp = get_Y_and_P(adata, gene_identifier_column_name = 'gene_name')


def shuffle_ynp():
	import random
	shuffled_ynp = {}
	for df_name, df in ynp.items():
		rs = []
		for l in df.shape:
			rs.append(np.array(random.sample(population = range(l), k = l)))
		rrows, rcols = rs
		shuffled = pd.DataFrame(
			data = np.array(df)[rrows[:, np.newaxis], rcols],
			columns = df.columns,
			index = df.index)
		shuffled_ynp[df_name] = shuffled
	return(shuffled_ynp)

data_sources = {'regular': ynp}
for ii in range(3):
	dn = f'shuffled_{ii}'
	data_sources[dn] = shuffle_ynp()


estimated_networks = {}
for data_source, data in data_sources.items():
	# data_source, data = list(data_sources.items())[0]

	P = data['P']
	Y = data['Y']

	# Get expression relative to controls.
	# The lsco* methods expect that.
	nt = 'non-targeting'
	control_expression = Y.loc[nt, :].mean(axis = 0)
	log2_fold_changes = np.log2(Y / control_expression)
	nt_bool = (Y.index != nt)
	log2_fold_changes = log2_fold_changes.loc[nt_bool, :]
	P_no_control = P.loc[nt_bool, :]

	Y = log2_fold_changes
	P = P_no_control

	m = 'lsco'
	try:
		en = gs.inference.infer_networks(
			Y = Y, P = P, 
			method = m)
		estimated_networks[f'{m}_{data_source}'] = en
	except Exception as e:
		print(f'{m} failed with:')
		print(e)

	m = 'zscore_ab'
	en = gs.inference.infer_networks(
		Y = Y, P = P, 
		method = m)
	en[np.isnan(en)] = 0
	estimated_networks[f'{m}_{data_source}'] = en

	# In [26]: np.isnan(estimated_networks['zscore']).sum().sum() / math.prod(estimated_networks['
	#     ...: zscore'].shape)
	# Out[26]: 0.711155952724128

	m = 'zscore_dream3'
	en = gs.inference.infer_networks(
		Y = Y, P = P, 
		method = m)
	en[np.isnan(en)] = 0
	estimated_networks[f'{m}_{data_source}'] = en

anton_util.pickle_object(
	estimated_networks, 
	output_dir / f'estimated_networks.pkl')



