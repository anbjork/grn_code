import numpy as np
import pandas as pd
import anndata as ad

from pathlib import Path
import copy

import anton_util

import functions

data_set_name = 'K562_essential_raw_singlecell_01'
anton_util.log_timestamp(f'Loading {data_set_name}...')

# For bulk data
# pseudo_bulk_path = Path(f'data/replogle/{data_set_name}_pseudo_bulk.h5ad')
# adata = ad.read_h5ad(pseudo_bulk_path)

preprocessed_path = Path(f'data/replogle/{data_set_name}_preprocessed.h5ad')
# Debug, use subset for speed
# preprocessed_path = Path(f'data/replogle/{data_set_name}_preprocessed_subset.h5ad')

adata = ad.read_h5ad(preprocessed_path)



anton_util.log_timestamp('Building Y and P...')
Y = functions.Y_from_adata(adata)
ynp = functions.get_Y_and_P(Y)

datasets = [ynp]




# anton_util.log_timestamp('shuffling...')
# updated = []
# for dataset in datasets:
#     updated.append({
#             'shuffle': False,
#             **dataset,
#             })
#     for jj in range(1):
#         ynp = {k: v for k, v in dataset.items() if k in ['Y', 'P']}
#         ynp = functions.shuffle_ynp(ynp)
#         simulation_shuffled = copy.deepcopy(dataset)
#         simulation_shuffled.update(ynp)
#         updated.append({
#             'shuffle': jj,
#             **simulation_shuffled,
#         })
# datasets = updated




anton_util.log_timestamp('extracting controls...')
for ii, dataset in enumerate(datasets):
    anton_util.log_timestamp(f'dataset {ii}...')
    Y = dataset['Y']
    nt = 'non-targeting'
    nt_bool = (Y.index == nt)
    control_cells = Y.loc[nt_bool, :]
    dataset['control_cells'] = control_cells






anton_util.log_timestamp('pseudo bulking...')
updated = []
for ii, dataset in enumerate(datasets):
    anton_util.log_timestamp(f'dataset {ii}...')
    updated.append({
        'pseudo_bulk': False,
        **dataset,
    })
    n_pseudo_bulk_options = [1, 2, 3, 5, 10]

    # Debug versions
    # n_pseudo_bulk_options = [5]
    # n_pseudo_bulk_options = [2, 5, 10]

    for n_pseudo_bulks in n_pseudo_bulk_options:
        anton_util.log_timestamp(f'n_pseudo_bulks: {n_pseudo_bulks}...')
        mats = {k: v for k, v in dataset.items() if k in ['Y']}
        P_bulk, pseudo_bulks = functions.bin_bulk(
            P = dataset['P'],
            matrices = mats,
            n_pseudo_bulks = n_pseudo_bulks,
            verbose = True,
        )
        pseudo_bulks['P'] = P_bulk
        data_updated = copy.deepcopy(dataset)
        data_updated.update(pseudo_bulks)
        updated.append({
            'pseudo_bulk': n_pseudo_bulks,
            **data_updated,
        })
datasets = updated









anton_util.log_timestamp('Computing log fold changes...')
for ii, dataset in enumerate(datasets):
    anton_util.log_timestamp(f'dataset {ii}...')

    Y = dataset['Y']
    P = dataset['P']
    control_cells = dataset['control_cells']

    nt = 'non-targeting'

    # Old version, before separating the controls before pseudo bulking
    # Remove when verified
    # nt_bool = (Y.index == nt)
    # P_no_control = P.loc[~nt_bool, :]
    # control = np.log2(Y.loc[nt_bool, :].values + 1).mean(axis=0)
    # log2_fold_changes = pd.DataFrame(
    #     np.log2(Y.loc[~nt_bool, :].values + 1) - control,
    #     index=Y.loc[~nt_bool, :].index,
    #     columns=Y.columns
    # )
    control = np.log2(control_cells.values + 1).mean(axis=0)
    log2_fold_changes = pd.DataFrame(
        np.log2(Y.values + 1) - control,
        index=Y.index,
        columns=Y.columns
    )

    # if dataset['pseudo_bulk'] is not False:
    #     haha

    dataset['log_fold_changes'] = {
        'Y': log2_fold_changes,
        # 'P': P_no_control,
        'P': P,
        }


# Debug, for speedy inference
# datasets = datasets[1:]


anton_util.log_timestamp('saving...')
outdir = Path('data/replogle')
outdir.mkdir(exist_ok=True, parents=True)
anton_util.pickle_object(datasets, f'{outdir}/{data_set_name}_preprocessed_2.pkl')










