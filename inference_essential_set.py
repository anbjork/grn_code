import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
import genesnake as gs

output_dir = Path('inferences')
output_dir.mkdir(exist_ok=True)

data_set_name = 'K562_essential_raw_singlecell_01'
pseudo_bulk_path = Path(f'data/replogle/{data_set_name}_pseudo_bulk.h5ad')

adata = ad.read_h5ad(pseudo_bulk_path)

# sc.get.aggregate stores results in layers, not X
# Move the 'sum' layer to X
adata.X = adata.layers['sum']

def get_Y_and_P(adata, knockdown_value=-1):
	Y = adata.to_df()
	Y.index.name = 'perturbations'
	Y.columns = adata.var['gene_name']
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

	return {'Y': Y, 'P': P}


ynp = get_Y_and_P(adata)


def shuffle_ynp():
	import random
	shuffled_ynp = {}
	for df_name, df in ynp.items():
		rs = []
		for l in df.shape:
			rs.append(np.array(random.sample(population=range(l), k=l)))
		rrows, rcols = rs
		shuffled = pd.DataFrame(
			data=np.array(df)[rrows[:, np.newaxis], rcols],
			columns=df.columns,
			index=df.index)
		shuffled_ynp[df_name] = shuffled
	return shuffled_ynp


data_sources = {'regular': ynp}
for ii in range(3):
	dn = f'shuffled_{ii}'
	data_sources[dn] = shuffle_ynp()


estimated_networks = {}
for data_source, data in data_sources.items():

	P = data['P']
	Y = data['Y']


	# # Get expression relative to controls.
	# # The lsco* methods expect that.
	# nt = 'non-targeting'
	# control_expression = Y.loc[nt, :].mean(axis=0)
	# log2_fold_changes = np.log2(Y / control_expression)
	# nt_bool = (Y.index != nt)
	# log2_fold_changes = log2_fold_changes.loc[nt_bool, :]
	# P_no_control = P.loc[nt_bool, :]
    #
	# Y = log2_fold_changes
	# P = P_no_control

	m = 'lsco'
	try:
		en = gs.inference.infer_networks(
			Y=Y, P=P,
			method=m)
		estimated_networks[f'{m}_{data_source}'] = en
	except Exception as e:
		print(f'{m} failed with:')
		print(e)

	m = 'zscore_ab'
	en = gs.inference.infer_networks(
		Y=Y, P=P,
		method=m)
	en[np.isnan(en)] = 0
	estimated_networks[f'{m}_{data_source}'] = en

	m = 'zscore_dream3'
	en = gs.inference.infer_networks(
		Y=Y, P=P,
		method=m)
	en[np.isnan(en)] = 0
	estimated_networks[f'{m}_{data_source}'] = en

import anton_util
anton_util.pickle_object(
	estimated_networks,
	output_dir / f'estimated_networks.pkl')
