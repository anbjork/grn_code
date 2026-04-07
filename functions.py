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
        # anton_util.log_timestamp(f'{m} finished.')
    except Exception as e:
        print(f'{m} failed with:')
        print(e)


    m = 'lsco'
    m_name = f'{m}.T'
    anton_util.log_timestamp(f'Running {m_name}...')
    try:
        en = gs.inference.infer_networks(
            Y = data['log_fold_changes']['Y'],
            P = data['log_fold_changes']['P'],
            method=m)
        estimated_networks[m_name] = en.T
        # anton_util.log_timestamp(f'{m_name} finished.')
    except Exception as e:
        print(f'{m_name} failed with:')
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
    # anton_util.log_timestamp(f'{m} finished.')

    # print(en)

    m = 'zscore_dream3'
    anton_util.log_timestamp(f'Running {m}...')
    en = gs.inference.infer_networks(
        Y=Y, P=P,
        method=m)
    en[np.isnan(en)] = 0
    estimated_networks[m] = en
    # anton_util.log_timestamp(f'{m} finished.')

    return estimated_networks





def pseudo_bulk_group(Y, n_pseudo_bulks):

    from math import floor
    smallest_bin = 5
    chunk_size = floor(Y.shape[0] / n_pseudo_bulks)
    if chunk_size < smallest_bin:
        print(f'not enough cells for {smallest_bin} cells per bin')
        n_pseudo_bulks = floor(Y.shape[0] / smallest_bin)
        print(f'using {n_pseudo_bulks} pseudo bulks instead')
    chunk_indices = chunk_size * np.array(range(n_pseudo_bulks))
    chunks = []
    for ii in chunk_indices:
        chunk = Y.iloc[ii : ii + chunk_size, :]
        chunks.append(chunk)
    # Redo the last chunk to include the remainder.
    # A chunk that is slightly bigger at the end should be much
    # better than a small remainder chunk, for the statistical properties
    # of the pseudo bulks. Size difference should be negligible too
    chunks.pop()
    chunks.append(Y.iloc[chunk_indices[-1] : , :])
    #
    # Probably a smarter way, except that the last chunk is not handled.
    # Maybe inspiration for improvement
    # //AB
    # chunks = [
    #     Y[i : i + chunk_size, :]
    #     for i in range(0, Y.shape[0], chunk_size)
    #     ]

    tmp = [f'psb{i}' for i in range(n_pseudo_bulks)]
    pseudo_bulk = {l: list(chunk.mean(axis = 0)) for l, chunk in zip(tmp, chunks)}
    return pseudo_bulk








def pseudo_bulk(matrices, n_pseudo_bulks = 5):
    import pandas as pd

    # Based on initial testing, pseudo bulking (like this) is a
    # significant part of the runtime for the lsco method

    pseudo_bulks = {matrix: [] for matrix in matrices}
    P = matrices['P']
    for perturbed_gene in P:
        perturbed_cell_indices = np.nonzero(P[perturbed_gene])[0]
        if len(perturbed_cell_indices) == 0:
            print(f'Gene {perturbed_gene} has no perturbations, skipping.')
            continue
        for matrix in matrices:
            tmp = matrices[matrix].iloc[perturbed_cell_indices, :]
            tmp2 = pseudo_bulk_group(tmp, n_pseudo_bulks = n_pseudo_bulks)
            for _, psb in tmp2.items():
                pseudo_bulks[matrix].append(psb)

    dfs = {}
    for matrix in matrices:
        df = pd.DataFrame(pseudo_bulks[matrix])
        df.columns = P.columns
        dfs[matrix] = df

    return dfs










