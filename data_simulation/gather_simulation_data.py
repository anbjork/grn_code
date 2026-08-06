

from pathlib import Path
import anton_util
from concurrent.futures import ProcessPoolExecutor
import numpy as np 
import genesnake as gs
import anton_util


from functions import extract_matrix_from_matlab_hdf5






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





outdir = Path(f'outputs/')
outdir.mkdir(exist_ok = True, parents = True)
anton_util.pickle_object(
    simulations,
    outdir / 'gathered_simulations.pkl'
    )







# def default_job_handler(inference_function, method, parallel = True):
#
#     from functools import partial
#     infer_network_method = partial(inference_function, method = method)
#
#     max_workers = 5
#
#     simulation_specifications = anton_util.unpickle_object(
#             'outputs/simulation_specifications.pkl'
#             )
#     if not parallel:
#         inferred_networks = []
#         for spec in simulation_specifications:
#             inferred_networks.append(infer_network_method(spec))
#     else:
#         with ProcessPoolExecutor(max_workers=max_workers) as executor:
#             # The map command is lazy and returns an generator.
#             # The list makes it execute. Otherwise it tries to pickle the 
#             # generator below
#             inferred_networks = list(executor.map(
#                 infer_network_method, 
#                 simulation_specifications,
#                 ))
#
#     outdir = Path(f'outputs/inference/{method}')
#     outdir.mkdir(exist_ok = True, parents = True)
#     anton_util.pickle_object(
#         inferred_networks,
#         outdir / 'networks.pkl'
#         )
#
#
