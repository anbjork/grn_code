
# import warnings
# warnings.filterwarnings("error")

from pathlib import Path
import anton_util
from grn_code import functions



def update_datasets(
        datasets,
        update_function,
        function_options,
        function_kwargs = {},
        ):
    from copy import deepcopy
    anton_util.log_timestamp(f'{update_function}...')
    updated = []
    for ii, dataset in enumerate(datasets):
        function_kwargs['index'] = ii
        anton_util.log_timestamp(f'dataset {ii}...')
        for option in function_options:
            # In case the update function modifies in place
            dataset = deepcopy(dataset)
            anton_util.log_timestamp(f'{option}...')
            updated.append(update_function(dataset, option, **function_kwargs))
    return updated




def find_variable_columns(Y):
    stds = Y.std(axis=0)
    means = Y.abs().mean(axis=0)
    atol = 1e-10
    rtol = 1e-3
    valid_cols = stds > (atol + rtol * means)
    return valid_cols


def find_flat_datasets(datasets):
    flat_datasets = [
            not find_variable_columns(dataset['Y']).all() 
            for dataset in datasets]
    return flat_datasets




anton_util.log_timestamp('reading..')

from grn_code.pipeline_code import pipeline_base_path
pipeline_base_path.mkdir(parents = True, exist_ok = True)

config = anton_util.unpickle_object(pipeline_base_path / 'pipeline_configuration.pkl')
options = config['preprocessing_options']

data_raw = anton_util.unpickle_object(pipeline_base_path / 'simulations.pkl')
datasets = data_raw


# datasets = datasets[:1]  # Debug
# print(type(datasets))
#
# # Subset data for debugging
# for dataset in datasets:
#     dataset['Y'] = dataset['Y'].iloc[:, :10]
#     dataset['A'] = dataset['A'].iloc[:10, :10]


functions.record_dropout_fractions(datasets, 'before_filtering')

datasets = update_datasets(
        datasets = datasets,
        update_function = functions.scanpy_preprocess,
        function_options = [None],
        )

functions.record_dropout_fractions(datasets, 'after_filtering')





datasets = update_datasets(
        datasets = datasets,
        update_function = functions.shuffle_y,
        function_options = options['shuffle'],
        )



function_kwargs = {'meta_data_label': 'cell normalised'}
anton_util.log_timestamp(f'{function_kwargs = }')
datasets = update_datasets(
        datasets = datasets,
        update_function = functions.normalise,
        function_options = options['cell normalised'],
        function_kwargs = function_kwargs,
        )



datasets = update_datasets(
        datasets = datasets,
        update_function = functions.bin_bulk,
        function_options = options['pseudo_bulk'],
        function_kwargs = {'terse' : True},
        )




function_kwargs = {'meta_data_label': 'read normalised'}
anton_util.log_timestamp(f'{function_kwargs = }')
datasets = update_datasets(
        datasets = datasets,
        update_function = functions.normalise,
        function_options = options['read normalised'],
        function_kwargs = function_kwargs,
        )


function_kwargs = {'meta_data_label': 'transform 1'}
anton_util.log_timestamp(f'{function_kwargs = }')
datasets = update_datasets(
    datasets = datasets,
    update_function = functions.transform,
    function_options = options['transform 1'],
    function_kwargs = function_kwargs,
    )


# zscores are often calculated after log1p, not instead of, so
# separate those steps. Can reuse the transform function though.
function_kwargs = {'meta_data_label': 'transform 2'}
anton_util.log_timestamp(f'{function_kwargs = }')
datasets = update_datasets(
    datasets = datasets,
    update_function = functions.transform,
    function_options = options['transform 2'],
    function_kwargs = function_kwargs,
    )


# Checking before control deltas, since I know they can sometimes
# introduce 0 stds. If 0 stds before that, something is unexpected and
# I want to know.
flat_datasets_1 = find_flat_datasets(datasets)

datasets = update_datasets(
    datasets = datasets,
    update_function = functions.compute_differences, 
    function_options = options['compute differences'],
    )

for dataset in datasets:
    Y = dataset['Y']
    cols = find_variable_columns(Y)
    Y = Y.loc[:, cols]
    dataset['Y'] = Y


anton_util.log_timestamp('extracting P...')
for ii, dataset in enumerate(datasets):
    anton_util.log_timestamp(f'dataset {ii}...')
    dataset['P'] = functions.get_P(dataset['Y'])



outfile = Path(pipeline_base_path / 'preprocessed_data.pkl')
anton_util.log_timestamp('saving...')
anton_util.pickle_object(datasets, outfile)





###### A few sanity checks on pre processed data #########
# Doing this after saving, so that data remains for inspection if crash.

def throw_if_any_nan_or_inf(datasets):
    import numpy as np
    for dataset in datasets:
        Y = dataset['Y']
        if np.any(np.isnan(Y)):
            raise ValueError('NaN found in Y')
        if np.any(np.isinf(Y)):
            raise ValueError('inf found in Y')

throw_if_any_nan_or_inf(datasets)

flat_datasets = find_flat_datasets(datasets)
import numpy as np
for df in [flat_datasets, flat_datasets_1]:
    if any(np.array(df)):
        raise ValueError('0 stds found')

metas = [d['meta'] for d in datasets]
for meta in metas:
    for k, v in meta['dataset_parameters'].items():
        meta[k] = v
    meta.pop('dataset_parameters')
import pandas as pd
df = pd.DataFrame(metas)



def plot_counts(array):
    from collections  import Counter
    tmp = sorted(Counter(array).items())
    rows, counts = zip(*tmp)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(rows, counts)
    return fig

cols = [d['Y'].shape[1] for d in datasets]
rows = [d['Y'].shape[0] for d in datasets]
plot_dir = Path(f'{pipeline_base_path}/unsorted/')
plot_dir.mkdir(parents = True, exist_ok = True)
for name, var in zip(['observations', 'genes'], [rows, cols]):
    fig = plot_counts(var)
    fig.savefig(f'{pipeline_base_path}/unsorted/dataset_counts_against_number_of_{name}.png')



