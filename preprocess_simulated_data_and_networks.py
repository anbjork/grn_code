
import pandas as pd
from pathlib import Path
import anton_util
from grn_code import functions

anton_util.log_timestamp('loading...')
data_set_name = 'simulated'
path = Path(
        '/home/anbjork/projects/replogle_round_2/versions/2/replogle_round_2/data_simulation/outputs/gathered_simulations.pkl'
        )
# Debug, subset for speed
# path = Path(
#         'data/simulated/subset.pkl'
#         )

data_raw = anton_util.unpickle_object(path)
datasets = data_raw


reference_networks = []
for ii, d in enumerate(datasets):
    reference_networks.append({'meta': {'replicate': ii}, 'data': d['A']})
    d.pop('A')
outfile = Path('outputs__in_pipeline/simulated/reference_networks.pkl')
outfile.parent.mkdir(exist_ok=True, parents=True)
anton_util.pickle_object(reference_networks, outfile)



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


def basic_formatting(dataset, _, **kwargs):
    d = dataset
    Y = d['Y']
    Y = functions.merge_p_into_y(Y, d['P'])
    controls = d['SCC']
    controls.index = ['control'] * len(controls)
    all = pd.concat([controls, d['Y']], axis = 0)
    out = ({
        'meta': {
            'replicate': kwargs['index']
            },
        'Y': all,
        'A': d['A'],
        })
    return out

# because kwargs['index'] is assigned to replicate in basic_formatting,
# this update must happen first for the replicate to correspond to original
# dataset replicates
datasets = update_datasets(
    datasets = datasets,
    update_function = basic_formatting,
    function_options = [None],
    )


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



# options = [False, True]
# options = [False]
options = [True]
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
n_pseudo_bulk_options = [False]
# n_pseudo_bulk_options = [False, 10]
datasets = update_datasets(
        datasets = datasets,
        update_function = functions.bin_bulk,
        function_options = n_pseudo_bulk_options,
        )




# options = [False, True]
# options = [True]
options = [False]
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



# options = [False, True]
options = [False]
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



outfile = Path('outputs__in_pipeline/simulated/data_processed.pkl')
outfile.parent.mkdir(exist_ok=True, parents=True)
anton_util.log_timestamp('saving...')
anton_util.pickle_object(datasets, outfile)






