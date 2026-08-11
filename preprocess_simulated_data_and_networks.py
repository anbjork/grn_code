
import pandas as pd
from pathlib import Path
import anton_util
from grn_code import functions

anton_util.log_timestamp('loading...')
data_set_name = 'simulated'
input_path = Path('src/grn_code/data_simulation/outputs/')
# Debug, subset for speed
# path = Path(
#         'data/simulated/subset.pkl'
#         )
output_path = Path('outputs__in_pipeline/simulated/')
output_path.mkdir(parents = True, exist_ok = True)

import shutil
shutil.copy(
    input_path / 'reference_networks.pkl',
    Path(output_path / 'reference_networks.pkl')
    )


def update_datasets(
        datasets,
        update_function,
        function_options,
        function_kwargs = {},
        ):
    anton_util.log_timestamp(f'{update_function}...')
    updated = []
    for ii, dataset in enumerate(datasets):
        function_kwargs['index'] = ii
        anton_util.log_timestamp(f'dataset {ii}...')
        for option in function_options:
            anton_util.log_timestamp(f'{option}...')
            updated.append(update_function(dataset, option, **function_kwargs))
    return updated



data_raw = anton_util.unpickle_object(input_path / 'simulations.pkl')
datasets = data_raw


# datasets = datasets[:1]  # Debug
# print(type(datasets))
#
# # Subset data for debugging
# for dataset in datasets:
#     dataset['Y'] = dataset['Y'].iloc[:, :10]
#     dataset['A'] = dataset['A'].iloc[:10, :10]


for d in datasets:
    d['meta']['0_fraction'] = functions.calculate_zero_fraction(d['Y'])




for dataset in datasets:
    Y = dataset['Y']
    stds = Y.std(axis = 0)
    print('0 stds:')
    print(sum(stds == 0))
    # Hmm, comparing floats to 0
    # Not optimal. Let's see though
    Y = Y.loc[:, stds > 0]
    dataset['Y'] = Y
    print(Y.shape)



datasets = update_datasets(
        datasets = datasets,
        update_function = functions.shuffle_y,
        # function_options = [False, True],
        function_options = [False],
        )



options = [False, True]
# options = [False]
# options = [True]
function_kwargs = {'meta_data_label': 'cell normalised'}
anton_util.log_timestamp(f'{function_kwargs = }')
datasets = update_datasets(
        datasets = datasets,
        update_function = functions.normalise,
        function_options = options,
        function_kwargs = function_kwargs,
        )



# n_pseudo_bulk_options = [False, 1, 2, 3, 5, 10]
# # Debug versions
# n_pseudo_bulk_options = [10]
# n_pseudo_bulk_options = [False]
n_pseudo_bulk_options = [False, 10]
datasets = update_datasets(
        datasets = datasets,
        update_function = functions.bin_bulk,
        function_options = n_pseudo_bulk_options,
        )




options = [False, True]
# options = [True]
# options = [False]
function_kwargs = {'meta_data_label': 'read normalised'}
anton_util.log_timestamp(f'{function_kwargs = }')
datasets = update_datasets(
        datasets = datasets,
        update_function = functions.normalise,
        function_options = options,
        function_kwargs = function_kwargs,
        )



# transforms = ['none', 'log1p', 'zscores']
# Debug versions
transforms = ['log1p']
# transforms = ['none', 'log1p']
function_kwargs = {'meta_data_label': 'transform 1'}
anton_util.log_timestamp(f'{function_kwargs = }')
datasets = update_datasets(
    datasets = datasets,
    update_function = functions.transform,
    function_options = transforms,
    function_kwargs = function_kwargs,
    )

from copy import deepcopy

bss = deepcopy(datasets)


# zscores are often calculated after log1p, not instead of, so
# separate those steps. Can reuse the transform function though.
# transforms = ['none', 'log1p', 'zscores']
# Debug versions
# transforms = ['zscores']
transforms = ['none', 'zscores']
# transforms = ['none']
function_kwargs = {'meta_data_label': 'transform 2'}
anton_util.log_timestamp(f'{function_kwargs = }')
datasets = update_datasets(
    datasets = datasets,
    update_function = functions.transform,
    function_options = transforms,
    function_kwargs = function_kwargs,
    )


css = deepcopy(datasets)



options = [False, True]
# options = [False]
datasets = update_datasets(
    datasets = datasets, 
    update_function = functions.compute_differences, 
    function_options = options)


dss = deepcopy(datasets)



anton_util.log_timestamp('extracting P...')
for ii, dataset in enumerate(datasets):
    anton_util.log_timestamp(f'dataset {ii}...')
    dataset['P'] = functions.get_P(dataset['Y'])



import numpy as np
for dataset in datasets:
    Y = dataset['Y']
    # print(dataset['meta'])
    dataset['meta']['any nan'] = np.any(np.isnan(Y))
metas = [d['meta'] for d in datasets]
# For debugging and manual inspection, not saved
df = pd.DataFrame(metas)


outfile = Path(output_path / 'data_processed.pkl')
anton_util.log_timestamp('saving...')
anton_util.pickle_object(datasets, outfile)






