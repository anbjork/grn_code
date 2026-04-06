import numpy as np
import pandas as pd
import anndata as ad

from pathlib import Path
import copy

import anton_util

import functions

data_set_name = 'K562_essential_raw_singlecell_01'
anton_util.log_timestamp(f'Loading {data_set_name}...')
# pseudo_bulk_path = Path(f'data/replogle/{data_set_name}_pseudo_bulk.h5ad')
# adata = ad.read_h5ad(pseudo_bulk_path)
preprocessed_path = Path(f'data/replogle/{data_set_name}_preprocessed.h5ad')
adata = ad.read_h5ad(preprocessed_path)

anton_util.log_timestamp('Building Y and P...')
Y = functions.Y_from_adata(adata)
ynp = functions.get_Y_and_P(Y)
anton_util.log_timestamp('Y and P built.')

dataset = ynp

with_shuffles = []
with_shuffles.append({
        'shuffle': False,
        **dataset,
        })
for jj in range(1):
    ynp = {k: v for k, v in dataset.items() if k in ['Y', 'P']}
    ynp = functions.shuffle_ynp(ynp)
    simulation_shuffled = copy.deepcopy(dataset)
    simulation_shuffled.update(ynp)
    with_shuffles.append({
        'shuffle': jj,
        **simulation_shuffled,
    })


data_out = []
for elem in with_shuffles:

    Y = elem['Y']
    P = elem['P']

    nt = 'non-targeting'
    nt_bool = (Y.index == nt)
    P_no_control = P.loc[~nt_bool, :]
    anton_util.log_timestamp('Computing log fold changes...')
    control = np.log2(Y.loc[nt_bool, :].values + 1).mean(axis=0)
    log2_fold_changes = pd.DataFrame(
        np.log2(Y.loc[~nt_bool, :].values + 1) - control,
        index=Y.loc[~nt_bool, :].index,
        columns=Y.columns
    )
    anton_util.log_timestamp('Log fold changes computed.')

    # SCC = elem['SCC']
    # log2_fold_changes = np.log2(Y + 1) - np.log2(SCC + 1)

    elem['log_fold_changes'] = {
        'Y': log2_fold_changes,
        'P': P_no_control,
        }
    data_out.append(elem)

outdir = Path('data/replogle')
outdir.mkdir(exist_ok=True, parents=True)
anton_util.pickle_object(data_out, f'{outdir}/{data_set_name}_preprocessed_2.pkl')










