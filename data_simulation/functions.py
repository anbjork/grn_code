import h5py
import pandas as pd


def extract_matrix_from_matlab_hdf5(file_path):
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



