import anndata as ad
from pathlib import Path
import anton_util



anton_util.log_timestamp('loading...')
data_set_name = 'simulated'
path = Path(
        '/home/anbjork/projects/replogle_round_2/versions/2/replogle_round_2/simulate_data/simulated_single_cell_benchmark/data_cases/outputs/gathered_simulations.pkl'
        )
data = anton_util.unpickle_object(path)
# get_Y_and_P uses the row names of Y, but they are not set on the
# simulated data, so it doesn't work as expected for those.
# One could imagine reconstructing the row names and running get_Y_and_P
# to verify that the function works. But we already know it does that,
# since we have better than 0 performance for real data,
# so won't do that for now
# //AB
#
# data_original = copy.deepcopy(data_raw)
# data = {}
# for ii, simulation in enumerate(data_raw):
#     anton_util.log_timestamp('Building Y and P...')
#     ynp = functions.get_Y_and_P(simulation['Y'])
#     assert((ynp['P'] == simulation['P']).all().all())
#     anton_util.log_timestamp('Y and P built.')
#     simulation.update(ynp)
#     data[f'simulated_{ii}'] = simulation


anton_util.log_timestamp('subsetting...')
for ii, dataset in enumerate(data):
    anton_util.log_timestamp(f'{ii}...')
    Y = dataset['Y']
    n = dataset['Y'].shape[0]
    import random
    iis = random.sample(range(n), int(n * 0.1))
    matnames = ['Y', 'P', 'SCC']
    for matrix_name in matnames:
        mat = dataset[matrix_name]
        mat = mat.iloc[iis, :]
        mat.reset_index(drop = True, inplace=True)

anton_util.log_timestamp('saving...')
outdir = Path('data/simulated')
outdir.mkdir(exist_ok=True, parents=True)
anton_util.pickle_object(data, f'{outdir}/subset.pkl')




