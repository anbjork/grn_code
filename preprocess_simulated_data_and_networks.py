
import numpy as np
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
for d in datasets:
    d.update({'0_fraction': functions.calculate_zero_fraction(d['Y'])})

# datasets = datasets[:2]  # Debug


anton_util.log_timestamp('shuffling...')
updated = []
for ii, simulation in enumerate(datasets):
    updated.append({
        'index': ii,
        'shuffle': False,
        **simulation,
    })
    for jj in range(1):
        ynp = {k: v for k, v in simulation.items() if k in ['Y', 'P']}
        ynp = functions.shuffle_ynp(ynp)
        simulation_shuffled = copy.deepcopy(simulation)
        simulation_shuffled.update(ynp)
        updated.append({
            'index': ii,
            'shuffle': jj,
            **simulation_shuffled,
        })
datasets = updated

# datasets = datasets[:1]  # Debug


anton_util.log_timestamp('pseudo bulking...')
updated = []
for ii, simulation in enumerate(datasets):
    anton_util.log_timestamp(f'{ii}...')
    updated.append({
        'pseudo_bulk': False,
        **simulation,
    })
    n_pseudo_bulk_options = [1, 2, 3, 5, 10]
    for n_pseudo_bulks in n_pseudo_bulk_options:
        anton_util.log_timestamp(f'{n_pseudo_bulks} pseudo bulks...')
        mats = {k: v for k, v in simulation.items() if k in ['Y', 'P', 'SCC']}
        P_bulk, pseudo_bulks = functions.bin_bulk(
            P = simulation['P'],
            matrices = mats,
            n_pseudo_bulks = n_pseudo_bulks,
            verbose = True,
        )
        pseudo_bulks['P'] = P_bulk
        data_updated = copy.deepcopy(simulation)
        data_updated.update(pseudo_bulks)
        updated.append({
            'pseudo_bulk': n_pseudo_bulks,
            **data_updated,
        })
datasets = updated






anton_util.log_timestamp('calculating log fold changes...')
for elem in datasets:

    Y = elem['Y']
    P = elem['P']
    SCC = elem['SCC']

    log2_fold_changes = np.log2(Y + 1) - np.log2(SCC + 1)

    elem['log_fold_changes'] = {
        'Y': log2_fold_changes,
        'P': P,
        }





anton_util.log_timestamp('saving...')
outdir = Path('data/simulated')
outdir.mkdir(exist_ok=True, parents=True)
anton_util.pickle_object(datasets, f'{outdir}/preprocessed.pkl')





