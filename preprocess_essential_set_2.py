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
nt = 'non-targeting'
# Giving the the controls a generic dataset agnostic name
tmp = np.array(df.index)
tmp[tmp == nt] = 'controls'
df.index = tmp
dropouts = functions.calculate_zero_fraction(df)
datasets = [{
    'meta': {
        '0_fraction': dropouts,
        },
    'Y': df,
    }]



# Might not be updated
#
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
    ds = copy.deepcopy(dataset)
    ds['meta']['pseudo_bulk'] = False
    updated.append(ds)
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
        data_updated['meta']['pseudo_bulk'] = n_pseudo_bulks
        updated.append(data_updated)
datasets = updated



anton_util.log_timestamp('transforming...')
updated = []
for ii, dataset in enumerate(datasets):
    anton_util.log_timestamp(f'dataset {ii}...')
    ds = copy.deepcopy(dataset)
    ds['meta']['transform'] = 'raw'
    updated.append(ds)

    transforms = ['log1p', 'zscores']
    for transform in transforms:
        anton_util.log_timestamp(f'transform {transform}...')
        transformed = functions.transform(dataset['Y'], transform)

        ds = copy.deepcopy(dataset)
        ds['meta']['transform'] = transform
        ds['Y'] = transformed
        updated.append(ds)
datasets = updated





updated = []
anton_util.log_timestamp('Computing differences...')
for ii, dataset in enumerate(datasets):
    anton_util.log_timestamp(f'dataset {ii}...')

    ds = copy.deepcopy(dataset)
    ds['meta']['delta'] = False
    updated.append(ds)

    Y = dataset['Y']
    ct_bool = (Y.index == 'controls')
    control_cells = Y.loc[ct_bool, :]
    control = control_cells.mean(axis=0)
    Y = Y.loc[~ct_bool, :]

    delta = pd.DataFrame(
        Y - control,
        index = Y.index,
        columns = Y.columns
    )

    ds = copy.deepcopy(dataset)
    ds['Y'] = delta
    ds['meta']['delta'] = True
    updated.append(ds)

datasets = updated






anton_util.log_timestamp('extracting P...')
for ii, dataset in enumerate(datasets):
    anton_util.log_timestamp(f'dataset {ii}...')
    dataset['P'] = functions.get_P(dataset['Y'])



# Debug, for speedy inference
datasets = [d for d in datasets if d['meta']['pseudo_bulk'] is not False]



anton_util.log_timestamp('saving...')
outdir = Path('data/replogle')
outdir.mkdir(exist_ok=True, parents=True)
anton_util.pickle_object(datasets, f'{outdir}/{data_set_name}_preprocessed_2.pkl')





