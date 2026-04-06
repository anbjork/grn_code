import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import genesnake as gs
import pickle
from pathlib import Path
from copy import deepcopy
import anton_util




def Y_from_adata(adata):
    Y = adata.to_df()
    Y.index = adata.obs['gene']
    Y.index.name = 'perturbations'
    Y.columns = adata.var['gene_name']
    Y.columns.name = 'genes'
    return Y


def shuffle_ynp(ynp):
    import random
    shuffled_ynp = {}
    for df_name, df in ynp.items():
        rs = []
        for l in df.shape:
            rs.append(np.array(random.sample(population=range(l), k=l)))
        rrows, rcols = rs
        shuffled = pd.DataFrame(
            data=np.array(df)[rrows[:, np.newaxis], rcols],
            columns=df.columns,
            index=df.index)
        shuffled_ynp[df_name] = shuffled
    return shuffled_ynp


def get_Y_and_P(Y, knockdown_value=-1):
    rows = np.array(Y.index)
    cols = np.array(Y.columns)
    M, N = Y.shape
    P_np = np.zeros(Y.shape)
    for ii in range(M):
        for jj in range(N):
            if rows[ii] == cols[jj]:
                P_np[ii, jj] = knockdown_value

    P = pd.DataFrame(P_np)
    P.index = rows
    P.index.name = 'perturbations'
    P.columns = cols
    P.columns.name = 'genes'

    return {'Y': Y, 'P': P}



def benchmark_method_against_reference(
    method,
    estimated_network,
    ref_name,
    reference_network,
    benchmark_output_dir,
    ):

    filter_reference = True
    if filter_reference:
        conditions = []
        for axis in range(len(reference_network.shape)):
            conditions.append(np.isin(
                np.array(reference_network.axes[axis]),
                np.array(estimated_network.axes[axis])
                ))
        reference_network = (
            reference_network.loc[tuple(conditions)]
            )
    else:
        reference_network = reference_network

    tmp = gs.util.harmonise_networks((
        estimated_network,
        reference_network))
    harmonised_estimated_network, harmonised_reference_network = tmp

    # Plots overwrite each other, because the assumptions of this function
    # does not match the file path structure of the pipeline.
    # Pipeline code is typically the outer code, it tends to change
    # from time to another to adapt to the structure of the results/project,
    # and it is usually the code handling the file paths. This function
    # does not need and should not need all metadata about what it is
    # benchmarking against what. So, the solution is to rewrite this
    # function to return the plots instead, so that the outer pipeline code
    # can attach meta data and save to disk as suitable for the case.
    # Won't fix right now, but should come back to do it. Took a todo on that.
    tmp = gs.benchmarking.benchmark(
        estimated_network=harmonised_estimated_network,
        reference_network=harmonised_reference_network.astype(bool),
        plot_dir=benchmark_output_dir / method,
        method_name=method,
        )
    ntps = np.nonzero(harmonised_reference_network)[0].shape[0]
    n_genes_after_harmonisation = harmonised_reference_network.shape[0]
    stats = {
        'method': method,
        'reference_network': ref_name,
        'n_TPs': ntps,
        'n_genes_after_harmonisation': n_genes_after_harmonisation,
        **tmp
        }

    return stats








def run_inference_on_data(data):

    estimated_networks = {}

    m = 'lsco'
    anton_util.log_timestamp(f'Running {m}...')
    try:
        en = gs.inference.infer_networks(
            Y = data['log_fold_changes']['Y'],
            P = data['log_fold_changes']['P'],
            method=m)
        estimated_networks[m] = en
        anton_util.log_timestamp(f'{m} finished.')
    except Exception as e:
        print(f'{m} failed with:')
        print(e)



    P = data['P']
    Y = data['Y']

    m = 'zscore_ab'
    anton_util.log_timestamp(f'Running {m}...')
    en = gs.inference.infer_networks(
        Y=Y, P=P,
        method=m)
    en[np.isnan(en)] = 0
    estimated_networks[m] = en
    anton_util.log_timestamp(f'{m} finished.')

    print(en)

    m = 'zscore_dream3'
    anton_util.log_timestamp(f'Running {m}...')
    en = gs.inference.infer_networks(
        Y=Y, P=P,
        method=m)
    en[np.isnan(en)] = 0
    estimated_networks[m] = en
    anton_util.log_timestamp(f'{m} finished.')

    return estimated_networks



