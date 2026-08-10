


import pandas as pd
import genesnake as gs
import json
import sys
import h5py
from pathlib import Path


input_file = sys.argv[1]
with open(input_file, 'r') as f:
    chunk = json.load(f)


for job_specification in chunk:

    pgs = 'python_global_parameters_for_matlab'
    sparsity = job_specification[pgs]['average_network_degree']
    genes = job_specification[pgs]['number_of_genes']

    model_type = 'lsco'
    # The make_scalefree network generation does not use the self_loops argument
    # self_loops = True
    # cell_replicates = job_specification['parameters']['cell_count']
    # 125 cell replicates is wayy much for genesnake to finish in reasonable time
    cell_replicates = 3
    snr = job_specification['parameters']['snr']
    noise_model = 'normal'


    network = gs.grn.make_scalefree(
        N = genes,
        S = sparsity
        # self_loops = self_loops
        )
    print('network generated')

    M = gs.GRNmodel.make_model(network)
    M.set_pert(
        'diag', effect = (0.9, 1), noise = 0.01, 
        reps = cell_replicates,
        )
    print('model generated')

    M.simulate_data(exp_type = 'ss', SNR = snr, noise_model = noise_model)
    print('data simulated')


    noise = M.noise
    data = M.data
    noise_free_data = M.noise_free_data
    reference_network = M.network
    assert((network == reference_network).all().all())
    SCC = pd.DataFrame([M.steady_state_RNA] * genes * cell_replicates).T
    SCC.columns = data.columns

    available_matrices = {
        # .T: Genespider and genesnake have differnt conventions for
        # the networks, so adjusting for that here, since this
        # genesnake version mimics the genespider one, to not require
        # a bunch of changes to later steps of the pipeline.
        'A': reference_network.T,
        'Y': data,
        'genesnake_noise_free': noise_free_data,
        'P': M.perturbation,
        'genesnake_noise': noise,
        'SCC': SCC,
    }

    matrix_files = job_specification['simulation_matrix_files']

    for name, data_matrix in available_matrices.items():
        if name in matrix_files:
            filename = matrix_files[name]
            # Remove existing file if it exists
            if Path(filename).exists():
                Path(filename).unlink()
            # Write to HDF5 file
            with h5py.File(filename, 'w') as f:
                # T: Soo, normally in this pipline, the hdf5 transfers between
                # matlab and python, which introduces a transpose,
                # see comment in the function that extracts from the hdf5 files.
                # Now, I am building a replacement simulation in Genesnake,
                # so now the hdf5 transfers from python to python, which means
                # no transpose. I am trying to change as little except the
                # simulation as possible, so transposing on writing here,
                # to simulate the transpose that the rest of the pipeline
                # expects.
                f.create_dataset('/data', data=data_matrix.T)
    
    # Create flag file to indicate completion
    flag_file = job_specification['simulation_completed_flag_file']
    Path(flag_file).touch()





