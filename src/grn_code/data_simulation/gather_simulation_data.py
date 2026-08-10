
from pathlib import Path
import anton_util


def extract_matrix_from_matlab_hdf5(file_path):

    import h5py
    import pandas as pd

    # Note: 
    # hdf5 does not enforce interpretation of the data,
    # Matlab is column major, and Numpy is row major.
    # Hence, on Matlab -> hdf5 -> Python, matrices are transposed.
    # It feels insane, but apparently a design choice by hdf5.
    #
    # Because of this, no transpose is needed to convert from
    # Genespiders [genes/variables x cells/observations] format to
    # Genesnakes [cells/observations x genes/variables] format.
    #
    # Jesus, the time and energy I've spent keeping track of transpositions
    with h5py.File(file_path) as f: 
        raw = f['data'][:] # pyright: ignore[reportIndexIssue]
    simulated_gene_names = [f'G{ii}' for ii in range(raw.shape[1])]  # pyright: ignore[reportAttributeAccessIssue]
    df = pd.DataFrame(
        data = raw,
        columns = simulated_gene_names  # pyright: ignore[reportArgumentType]
    )
    return df




simulation_specifications = anton_util.unpickle_object(
        'outputs/simulation_specifications.pkl'
        )
simulations = []
for spec in simulation_specifications:

    simulation_files = spec['simulation_matrix_files']
    parameter_tag = spec['parameter_tag']
    print(parameter_tag, flush = True)

    # # Note: 
    # # hdf5 does not enforce interpretation of the data,
    # # Matlab is column major, and Numpy is row major.
    # # Hence, on Matlab -> hdf5 -> Python, matrices are transposed.
    # # It feels insane, but apparently a design choice by hdf5.
    # #
    # # Because of this, no transpose is needed to convert from
    # # Genespiders [genes/variables x cells/observations] format to
    # # Genesnakes [cells/observations x genes/variables] format.
    # #
    # # Jesus, the time and energy I've spent keeping track of transpositions
    output_matrices = {
            name: extract_matrix_from_matlab_hdf5(simulation_files[name]) 
            for name in ['A', 'Y', 'P', 'SCC']
            }
    output_matrices['A'].index = output_matrices['A'].columns

    # # zscore can run on either counts or fold changes
    # for_inference, _ = Y, SCC
    # #
    # # fold_changes = np.log2(Y + 1) - np.log2(SCC + 1)
    # # for_inference = fold_changes

    simulations.append(output_matrices)





# Cleanup when sure. REMOVE
# def basic_formatting(dataset, _, **kwargs):
# def basic_formatting(dataset):

import pandas as pd
import replogle_round_2.functions
updated = []
for ii, dataset in enumerate(simulations):

    d = dataset
    Y = d['Y']
    Y = replogle_round_2.functions.merge_p_into_y(Y, d['P'])
    controls = d['SCC']
    controls.index = ['control'] * len(controls)
    all = pd.concat([controls, d['Y']], axis = 0)

    out = ({
        'meta': {
            # Add simulation parameeters, FIX
            'replicate': ii,
            },
        'Y': all,
        'A': d['A'],
        })
    updated.append(out)

# # because kwargs['index'] is assigned to replicate in basic_formatting,
# # this update must happen first for the replicate to correspond to original
# # dataset replicates
# datasets = update_datasets(
#     datasets = datasets,
#     update_function = basic_formatting,
#     function_options = [None],
#     )



outdir = Path(f'outputs/')
outdir.mkdir(exist_ok = True, parents = True)
anton_util.pickle_object(
    simulations,
    outdir / 'gathered_simulations.pkl'
    )


