
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
from grn_code.pipeline_configuration import pipeline_base_path as output_path
output_path.mkdir(parents = True, exist_ok = True)



from grn_code.pipeline_configuration import preprocessing_options as options



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


# anton_util.log_timestamp('reading..')
data_raw = anton_util.unpickle_object(input_path / 'simulations.pkl')
# anton_util.log_timestamp('reading done')
datasets = data_raw


# datasets = datasets[:1]  # Debug
# print(type(datasets))
#
# # Subset data for debugging
# for dataset in datasets:
#     dataset['Y'] = dataset['Y'].iloc[:, :10]
#     dataset['A'] = dataset['A'].iloc[:10, :10]


functions.record_dropout_fractions(datasets, 'before_gene_filtering')




datasets = update_datasets(
        datasets = datasets,
        update_function = functions.scanpy_preprocess,
        function_options = [None],
        )
datasets = update_datasets(
        datasets = datasets,
        update_function = functions.differential_expression_gene_filtering,
        function_options = [None],
        )

# Leaving this in as a sanity check on the DE filtering for now,
# but can be removed unless I see something strange
for dataset in datasets:
    Y = dataset['Y']
    stds = Y.std(axis = 0)
    if (stds == 0).any():
        raise ValueError('0 stds found')
    # print('0 stds:')
    # print(sum(stds == 0))
    # Hmm, comparing floats to 0
    # Not optimal. Let's see though
    # Y = Y.loc[:, stds > 0]
    # dataset['Y'] = Y
    # print(Y.shape)


functions.record_dropout_fractions(datasets, 'after_gene_filtering')





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

from copy import deepcopy

bss = deepcopy(datasets)


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


css = deepcopy(datasets)



datasets = update_datasets(
    datasets = datasets, 
    update_function = functions.compute_differences, 
    function_options = options['compute differences'],
    )


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






