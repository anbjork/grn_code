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
preprocessed_path = Path(f'data/replogle/{data_set_name}_preprocessed_subset.h5ad')
print('using path:')
print(preprocessed_path)

adata = ad.read_h5ad(preprocessed_path)



df = pd.DataFrame(
        data = adata.X,
        index = adata.obs.gene,
        columns = adata.var.gene_name,
        )
dropouts = functions.calculate_zero_fraction(df)
datasets = [{
    'Y': df,
    '0_fraction': dropouts,
    }]




# anton_util.log_timestamp('shuffling...')
# updated = []
# for dataset in datasets:
#     updated.append({
#             'shuffle': false,
#             **dataset,
#             })
#     for jj in range(1):
#         ynp = {k: v for k, v in dataset.items() if k in ['y', 'p']}
#         ynp = functions.shuffle_ynp(ynp)
#         simulation_shuffled = copy.deepcopy(dataset)
#         simulation_shuffled.update(ynp)
#         updated.append({
#             'shuffle': jj,
#             **simulation_shuffled,
#         })
# datasets = updated






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
    n_pseudo_bulk_options = [5]
    # n_pseudo_bulk_options = [2, 5, 10]

    for n_pseudo_bulks in n_pseudo_bulk_options:
        anton_util.log_timestamp(f'n_pseudo_bulks: {n_pseudo_bulks}...')
        # mats = {k: v for k, v in dataset.items() if k in ['Y']}
        # P_bulk, pseudo_bulks = functions.bin_bulk(
        mat_bulk = functions.bin_bulk(
            mat = dataset['Y'],
            # P = dataset['P'],
            # matrices = mats,
            n_pseudo_bulks = n_pseudo_bulks,
            # verbose = True,
        )
        # pseudo_bulks['P'] = P_bulk

        # Could probably be simplified, but sticking to current structure for now
        data_updated = copy.deepcopy(dataset)
        data_updated.update({'Y': mat_bulk})
        updated.append({
            'pseudo_bulk': n_pseudo_bulks,
            **data_updated,
        })

datasets = updated









anton_util.log_timestamp('Computing log fold changes...')
for ii, dataset in enumerate(datasets):
    anton_util.log_timestamp(f'dataset {ii}...')

    Y = dataset['Y']
    # P = dataset['P']

    nt = 'non-targeting'
    nt_bool = (Y.index == nt)
    control_cells = Y.loc[nt_bool, :]
    Y = Y.loc[~nt_bool, :]
    # P = P.loc[~nt_bool, :]

    control = np.log2(control_cells.values + 1).mean(axis=0)
    log2_fold_changes = pd.DataFrame(
        np.log2(Y.values + 1) - control,
        index=Y.index,
        columns=Y.columns
    )

    dataset['Y'] = {
            'raw': dataset['Y'],
            'log_fold_changes': log2_fold_changes,
            }

    # dataset['log_fold_changes'] = {
    #     'Y': log2_fold_changes,
    #     'P': P,
    #     }



anton_util.log_timestamp('extracting P...')
for ii, dataset in enumerate(datasets):
    anton_util.log_timestamp(f'dataset {ii}...')
    P = {}
    for mname, mat in dataset['Y'].items():
        # Y = dataset['Y']
        P[mname] = functions.get_P(mat)
        dataset['P'] = P



# Debug, for speedy inference
# datasets = datasets[1:]


anton_util.log_timestamp('saving...')
outdir = Path('data/replogle')
outdir.mkdir(exist_ok=True, parents=True)
anton_util.pickle_object(datasets, f'{outdir}/{data_set_name}_preprocessed_2.pkl')










