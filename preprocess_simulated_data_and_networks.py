
import pandas as pd
from pathlib import Path
import anton_util
import functions
import copy

anton_util.log_timestamp('loading...')
data_set_name = 'simulated'
path = Path(
        '/home/anbjork/projects/replogle_round_2/versions/2/replogle_round_2/simulate_data/simulated_single_cell_benchmark/data_cases/outputs/gathered_simulations.pkl'
        )
# Debug, subset for speed
# path = Path(
#         'data/simulated/subset.pkl'
#         )

outdir = Path('data_processed/simulated')
outdir.mkdir(exist_ok=True, parents=True)


data_raw = anton_util.unpickle_object(path)
# get_Y_and_P uses the row names of Y, but they are not set on the
# simulated data, so it doesn't work as expected for those.
# One could imagine reconstructing the row names and running get_Y_and_P
# to verify that the function works. But we already know it does that,
# since we have better than 0 performance for real data,
# so won't do that for now
# //AB
#
# data_original = copy.deepcopy(data_raw)
# data_sources = {}
# for ii, simulation in enumerate(data_raw):
#     anton_util.log_timestamp('Building Y and P...')
#     ynp = functions.get_Y_and_P(simulation['Y'])
#     assert((ynp['P'] == simulation['P']).all().all())
#     anton_util.log_timestamp('Y and P built.')
#     simulation.update(ynp)
#     data_sources[f'simulated_{ii}'] = simulation
datasets = data_raw

# datasets = datasets[:3]  # Debug




ground_truths = {}
for ii, d in enumerate(datasets):
    ground_truths[ii] = d['A']
anton_util.pickle_object(ground_truths, 'data_processed/simulated/ground_truths.pkl')




updated = []
for ii, d in enumerate(datasets):
    Y = d['Y']
    Y = functions.merge_p_into_y(Y, d['P'])
    controls = d['SCC']
    controls.index = ['control'] * len(controls)
    all = pd.concat([controls, d['Y']], axis = 0)
    updated.append({
        'meta': {
            'replicate': ii
            },
        'Y': all,
        'A': d['A'],
    })
datasets = updated



# datasets = datasets[:1]  # Debug
# print(type(datasets))
#
# # Subset data for debugging
# for dataset in datasets:
#     dataset['Y'] = dataset['Y'].iloc[:, :10]
#     dataset['A'] = dataset['A'].iloc[:10, :10]



for d in datasets:
    d['meta']['0_fraction'] = functions.calculate_zero_fraction(d['Y'])


# datasets = datasets[:2]  # Debug


# anton_util.log_timestamp('shuffling...')
# updated = []
# for ii, dataset in enumerate(datasets):
#     ds = copy.deepcopy(dataset)
#     ds['meta']['shuffle'] = False
#     updated.append(ds)
#
#     for jj in range(1):
#         ds = copy.deepcopy(dataset)
#         ds['meta']['shuffle'] = jj
#         ds['Y'] = functions.shuffle_y(ds['Y'])
#         updated.append(ds)
# datasets = updated

# datasets = datasets[:1]  # Debug



anton_util.log_timestamp('pseudo bulking...')
updated = []
for ii, dataset in enumerate(datasets):
    anton_util.log_timestamp(f'dataset {ii}...')

    ds = copy.deepcopy(dataset)
    ds['meta']['pseudo_bulk'] = False
    updated.append(ds)

    # # n_pseudo_bulk_options = [1, 2, 3, 5, 10]
    # # # Debug versions
    # n_pseudo_bulk_options = [10]
    # # # n_pseudo_bulk_options = [2, 5, 10]
    #
    # for n_pseudo_bulks in n_pseudo_bulk_options:
    #     anton_util.log_timestamp(f'n_pseudo_bulks: {n_pseudo_bulks}...')
    #     mat_bulk = functions.bin_bulk(
    #         mat = dataset['Y'],
    #         n_pseudo_bulks = n_pseudo_bulks,
    #     )
    #
    #     ds = copy.deepcopy(dataset)
    #     ds['Y'] = mat_bulk
    #     ds['meta']['pseudo_bulk'] = n_pseudo_bulks
    #     updated.append(ds)

datasets = updated






anton_util.log_timestamp('transforming...')
updated = []
for ii, dataset in enumerate(datasets):
    anton_util.log_timestamp(f'dataset {ii}...')
    # ds = copy.deepcopy(dataset)
    # ds['meta']['transform'] = 'raw'
    # updated.append(ds)

    transforms = ['raw', 'log1p', 'zscores']
    # Debug versions
    transforms = ['log1p']

    for transform in transforms:
        anton_util.log_timestamp(f'transform {transform}...')
        transformed = functions.transform(dataset['Y'], transform)

        ds = copy.deepcopy(dataset)
        ds['meta']['transform'] = transform
        ds['Y'] = transformed
        updated.append(ds)
datasets = updated





# updated = []
# anton_util.log_timestamp('Computing differences...')
# for ii, dataset in enumerate(datasets):
#     anton_util.log_timestamp(f'dataset {ii}...')
#
#     ds = copy.deepcopy(dataset)
#     ds['meta']['control_delta'] = False
#     updated.append(ds)
#
#     Y = dataset['Y']
#     ct_bool = (Y.index == 'controls')
#     control_cells = Y.loc[ct_bool, :]
#     control = control_cells.mean(axis=0)
#     Y = Y.loc[~ct_bool, :]
#
#     delta = pd.DataFrame(
#         Y - control,
#         index = Y.index,
#         columns = Y.columns
#     )
#
#     ds = copy.deepcopy(dataset)
#     ds['Y'] = delta
#     ds['meta']['control_delta'] = True
#     updated.append(ds)
#
# datasets = updated






anton_util.log_timestamp('extracting P...')
for ii, dataset in enumerate(datasets):
    anton_util.log_timestamp(f'dataset {ii}...')
    dataset['P'] = functions.get_P(dataset['Y'])





anton_util.log_timestamp('saving...')
outfile = f'{outdir}/preprocessed.pkl'
if Path(outfile).exists():
    previous_data = anton_util.unpickle_object(outfile)
    datasets = previous_data + datasets
anton_util.log_timestamp(f'total datasets: {len(datasets)}')
anton_util.pickle_object(datasets, outfile)






