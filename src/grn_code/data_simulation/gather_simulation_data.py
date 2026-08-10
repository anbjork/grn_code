
from pathlib import Path
import anton_util
import pandas as pd

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


def get_simuation_matrices_from_specification(specification):

    simulation_files = specification['simulation_matrix_files']

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

    return output_matrices


from grn_code.functions import merge_p_into_y

simulation_specifications = anton_util.unpickle_object(
        'outputs/simulation_specifications.pkl'
        )
datasets = []
reference_networks = []
for specification in simulation_specifications:

    m = get_simuation_matrices_from_specification(specification)
    Y = m['Y']
    meta = {
            'dataset_parameters': {
                **specification['python_global_parameters_for_matlab'],
                **specification['parameters'],
                }
            }
    reference_networks.append({'meta': meta, 'data': m['A']})

    Y = merge_p_into_y(Y, m['P'])
    controls = m['SCC']
    controls.index = ['control'] * len(controls)
    all = pd.concat([controls, m['Y']], axis = 0)
    out = ({
        'meta': meta,
        'Y': all,
        })
    datasets.append(out)

outdir = Path(f'outputs/')
outdir.mkdir(exist_ok = True, parents = True)
anton_util.pickle_object(reference_networks, outdir / 'reference_networks.pkl')
anton_util.pickle_object(datasets, outdir / 'simulations.pkl')


