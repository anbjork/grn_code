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





# def pseudo_bulk_group(Y, n_pseudo_bulks, verbose = True):
#
#     complain = False
#     complain_more = False
#
#     import math
#     smallest_intended_bin = 5
#     intended_n_pseudo_bulks = n_pseudo_bulks
#     chunk_size = floor(Y.shape[0] / n_pseudo_bulks)
#     if chunk_size < smallest_intended_bin:
#         complain = True
#         n_pseudo_bulks = floor(Y.shape[0] / smallest_intended_bin)
#         if n_pseudo_bulks == 0:
#             complain_more = True
#             n_pseudo_bulks = 1
#         chunk_size = floor(Y.shape[0] / n_pseudo_bulks)
#         if verbose:
#             print(f'smallest bin size is {smallest_intended_bin}')
#             print(f'cells in group is {Y.shape[0]}')
#             print(f'not enough cells for {intended_n_pseudo_bulks} pseudo bulks')
#             print(f'using {n_pseudo_bulks} pseudo bulks instead')
#             print(f'set chunk size to {chunk_size}')
#             print('remaining cells go with the last chunk')
#             if complain_more:
#                 print(f'not enough cells for even 1 pseudo bulk of intended size, so using 1 pseudo bulk with all cells')
#             print()
#     chunk_indices = chunk_size * np.array(range(n_pseudo_bulks))
#     chunks = []
#     for ii in chunk_indices:
#         chunk = Y.iloc[ii : ii + chunk_size, :]
#         chunks.append(chunk)
#     # Redo the last chunk to include the remainder.
#     # A chunk that is slightly bigger at the end should be much
#     # better than a small remainder chunk, for the statistical properties
#     # of the pseudo bulks. Size difference should be negligible too
#     chunks.pop()
#     chunks.append(Y.iloc[chunk_indices[-1] : , :])
#     #
#     # Probably a smarter way, except that the last chunk is not handled.
#     # Maybe inspiration for improvement
#     # //AB
#     # chunks = [
#     #     Y[i : i + chunk_size, :]
#     #     for i in range(0, Y.shape[0], chunk_size)
#     #     ]
#
#     tmp = [f'psb{i}' for i in range(n_pseudo_bulks)]
#     pseudo_bulk = {l: list(chunk.mean(axis = 0)) for l, chunk in zip(tmp, chunks)}
#     return pseudo_bulk, complain






def chunk_array(arr, n_chunks):
    # Separated this mostly to avoid index ception, since the arrays
    # I'll be chunking here are arrays of indices
    import math
    # n_chunks = math.floor(len(arr) / chunk_size)
    chunk_size = math.floor(len(arr) / n_chunks)
    chunk_indices = chunk_size * np.array(range(n_chunks))
    chunks = []
    for ii in chunk_indices:
        chunk = arr[ii : ii + chunk_size]
        chunks.append(chunk)
    # Redo the last chunk to include the remainder.
    # A chunk that is slightly bigger at the end should be much
    # better than a small remainder chunk, for the statistical properties
    # of the pseudo bulks. Size difference should be small too
    chunks.pop()
    chunks.append(arr[chunk_indices[-1] : ])
    return chunks



def bin_bulk(P, matrices, n_pseudo_bulks = 5, verbose = False):
    import pandas as pd
    import math

    # Based on initial testing, pseudo bulking is a
    # significant part of the runtime for the lsco method

    # def bitch_about_dimensions(complain_more):
    #     if complain:
    #         if complain_more:
    #             print(f'not enough cells for even 1 pseudo bulk of intended size')


    # def fix_and_bitch_about_pseudo_bulk_dimensions(n_cells, complain):
    #     complain_more = False
    #     import math
    #     n_pseudo_bulks = math.floor(n_cells / smallest_intended_bin)
    #     if n_pseudo_bulks == 0:
    #         complain_more = True
    #         n_pseudo_bulks = 1
    #     chunk_size = math.floor(n_cells / n_pseudo_bulks)
    #     return chunk_size, n_pseudo_bulks

    smallest_intended_bin = 5
    n_requested_pseudo_bulks = n_pseudo_bulks

    max_complaints = 3
    complaints_count = 0
    complain = True
    # verbose_sub_function = True
    # have_stopped_warning = False

    P_bulk = []
    pseudo_bulk_indices = []
    for perturbed_gene in P:
        # If it wasn't for counting complaints, I would put all
        # the next paragraph away in a neat and tidy function.
        # Can do workarounds like having part of the logic here
        # and part in a function, but didn't find a way that felt good.
        # So just inlined all for now
        if (
                # == rather than >= makes it trigger exactly once
                # which is good. Otherwise it constantly complains that 
                # it has stopped complaining. Feels slightly unintuitive somehow
                complaints_count == max_complaints and
                # not have_stopped_warning and
                not verbose
                ):
            print()
            print(f'Reached warning count limit of {max_complaints}, so will stop warning. To print all warnings, call with verbose = True')
            print()
            complain = False
        perturbed_cell_indices = np.nonzero(P[perturbed_gene])[0]
        n_cells = len(perturbed_cell_indices)
        if n_cells == 0:
            complaints_count = complaints_count + 1
            print(f'Gene {perturbed_gene} has no perturbations, skipping.')
            continue
        # chunk_size = math.floor(n_cells / n_requested_pseudo_bulks)
        n_pseudo_bulks = math.floor(n_cells / smallest_intended_bin)
        if n_pseudo_bulks > n_requested_pseudo_bulks:
            n_pseudo_bulks = n_requested_pseudo_bulks
        elif n_pseudo_bulks < n_requested_pseudo_bulks:
            complaints_count = complaints_count + 1
            # n_pseudo_bulks = math.floor(n_cells / smallest_intended_bin)
            if complain:
                print()
                print(f'smallest intended bin size is {smallest_intended_bin}')
                print(f'cells in group is {n_cells}')
                print(f'not enough cells for {n_requested_pseudo_bulks} pseudo bulks')
            if n_pseudo_bulks == 0:
                n_pseudo_bulks = 1
                if complain:
                    print(f'not enough cells for even 1 pseudo bulk of intended size')
            if complain:
                print(f'using {n_pseudo_bulks} pseudo bulks instead')
                # print(f'set chunk size to {chunk_size}')
                print('remaining cells go with the last pseudo bulk')
            # verbose_sub_function = False
            # have_stopped_warning = True

            # if n_pseudo_bulks == 0:
            #     complain more
            #     n_pseudo_bulks = 1

        # chunk_size = math.floor(n_cells / n_requested_pseudo_bulks)
        # if chunk_size < smallest_intended_bin:
        #     complain
        #     n_pseudo_bulks  = math.floor(n_cells / smallest_intended_bin)
        #     if n_pseudo_bulks == 0:
        #         n_pseudo_bulks = 1
        #     chunk_size = math.floor(n_cells / n_pseudo_bulks)

        # if chunk_size < smallest_intended_bin:

        # ..indices[0] is arbitrary
        # The code assumes single perturbations, so all rows pointed to by the
        # indices should be identical
        #
        # Actually worth checking!
        # Not unthinkable that this or some other dataset will include
        # multi target perturbations, and if so we want to know
        #
        # Actually, the way P is extracted, that shouldn't happpen.
        # So such checks would need to be before extracting P
        #
        # Remember to comment this out when verified, it might add extra runtime
        first = P.iloc[perturbed_cell_indices[0], :]
        for ii in perturbed_cell_indices:
            if not (P.iloc[ii, :] == first).all():
                raise ValueError(f'Perturbed cell indices for gene {perturbed_gene} do not point to identical rows in P.')
        #
        # Ran it some times with check above. Never triggered, so assumptions
        # seem correct.

        for _ in range(n_pseudo_bulks):
            P_bulk.append(list(first))

        indices_chunks = chunk_array(perturbed_cell_indices, n_pseudo_bulks)
        pseudo_bulk_indices.append(indices_chunks)

    P_bulk_df = pd.DataFrame(P_bulk)
    P_bulk_df.columns = P.columns
    # def pseudo_bulk_group(Y, n_pseudo_bulks, verbose = True):

    pseudo_bulks = {}
    for matrix_name, matrix in matrices.items():
        matbulk = []
        for gene_pseudo_bulks in pseudo_bulk_indices:
            for index_array in gene_pseudo_bulks:
                matbulk.append(list(matrix.iloc[index_array, :].mean(axis = 0)))
        df = pd.DataFrame(matbulk)
        df.columns = matrix.columns
        pseudo_bulks[matrix_name] = df

        #     # tmp = matrices[matrix].iloc[perturbed_cell_indices, :]
        #     tmp = matrix.iloc[perturbed_cell_indices, :]
        #
        # tmp = [f'psb{i}' for i in range(n_pseudo_bulks)]
        # pseudo_bulk = {l: list(chunk.mean(axis = 0)) for l, chunk in zip(tmp, chunks)}
        # # return pseudo_bulk, complain
        #
        #     # tmp2, warn = pseudo_bulk_group(
        #     #     tmp, n_pseudo_bulks = n_pseudo_bulks, verbose = verbose_sub_function)
        #
        #     for _, psb in tmp2.items():
        #         pseudo_bulks[matrix].append(psb)



    return P_bulk_df, pseudo_bulks






def calculate_zero_fraction(df):
    non_zeros = np.nonzero(df)[0].shape[0]
    d = 1 - non_zeros / df.size
    return d



