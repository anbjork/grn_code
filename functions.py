import numpy as np
import pandas as pd
import genesnake as gs
from pathlib import Path
import anton_util




def Y_from_adata(adata):
    Y = adata.to_df()
    Y.index = adata.obs['gene']
    Y.index.name = 'perturbations'
    Y.columns = adata.var['gene_name']
    Y.columns.name = 'genes'
    return Y




def shuffle_y(y):
    import random
    import copy
    l = y.shape[0]
    riis = random.sample(population=range(l), k=l)
    out = copy.deepcopy(y)
    out.index = y.index[riis]
    return out





# def shuffle_ynp(ynp):
#     import random
#     shuffled_ynp = {}
#     for df_name, df in ynp.items():
#         rs = []
#         for l in df.shape:
#             rs.append(np.array(random.sample(population=range(l), k=l)))
#         rrows, rcols = rs
#         shuffled = pd.DataFrame(
#             data=np.array(df)[rrows[:, np.newaxis], rcols],
#             columns=df.columns,
#             index=df.index)
#         shuffled_ynp[df_name] = shuffled
#     return shuffled_ynp


def get_P(Y, knockdown_value=-1):
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

    return P



def benchmark_method_against_reference(
    method,
    estimated_network,
    # ref_name,
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
        # 'method': method,
        # 'reference_network': ref_name,
        'n_TPs': ntps,
        'n_genes_after_harmonisation': n_genes_after_harmonisation,
        **tmp
        }

    return stats



def correlation_inference(data):

    Y = data['Y']

    # Calculate pearson correlation between all pairs of columns
    estimated_network = Y.corr(method='pearson')
    estimated_network[np.isnan(estimated_network)] = 0

    return estimated_network



def run_inference_on_data(data):

    estimated_networks = {}

    P = data['P']
    Y = data['Y']

    m = 'lsco'
    anton_util.log_timestamp(f'Running {m}...')
    try:
        en = gs.inference.infer_networks(
            Y = Y,
            P = P,
            method=m)
        estimated_networks[m] = en
    except Exception as e:
        print(f'{m} failed with:')
        print(e)


    m = 'lsco'
    m_name = f'{m}.T'
    anton_util.log_timestamp(f'Running {m_name}...')
    try:
        en = gs.inference.infer_networks(
            Y = Y,
            P = P,
            method=m)
        estimated_networks[m_name] = en.T
    except Exception as e:
        print(f'{m_name} failed with:')
        print(e)



    m = 'zscore_ab'
    anton_util.log_timestamp(f'Running {m}...')
    en = gs.inference.infer_networks(
        Y=Y, P=P,
        method=m)
    en[np.isnan(en)] = 0
    estimated_networks[m] = en

    # print(en)

    m = 'zscore_dream3'
    anton_util.log_timestamp(f'Running {m}...')
    en = gs.inference.infer_networks(
        Y=Y, P=P,
        method=m)
    en[np.isnan(en)] = 0
    estimated_networks[m] = en



    m = 'mean_difference'
    anton_util.log_timestamp(f'Running {m}...')
    en = gs.inference.infer_networks(
        Y=Y, P=P,
        method=m)
    en[np.isnan(en)] = 0
    estimated_networks[m] = en



    m = 'mean_difference_controls_only'
    anton_util.log_timestamp(f'Running {m}...')
    en = gs.inference.infer_networks(
        Y=Y, P=P,
        method=m)
    en[np.isnan(en)] = 0
    estimated_networks[m] = en

    return estimated_networks







def dspin_inference(data):

    from dspin.dspin import DSPIN
    import anndata as ad

    # Recreating an anndata object to fit with the rest of the code
    # from previously
    y = data['Y']

    # # Debug
    # y = y.iloc[:, :50]

    adata = ad.AnnData(
            # to_numpy and commented out obs because anndata stresses
            # over non unique obs names, so letting it assign an integer range
            X = y.to_numpy(),
            # obs = pd.DataFrame(index = y.index),
            var = pd.DataFrame(index = y.columns),
            )
    adata.obs['sample_id'] = y.index
    adata.obs['batch'] = 'mock_batch'  # Assuming only 1 batch
    controls = (y.index == 'control')
    adata.obs['if_control'] = controls
    # This gene name column is probably not needed. Without the sample_id_key
    # argument, the network inference would probably pick the the sample ids
    # from the sample_id column.
    # This way, we have both and are explicit about it, so keeping for now
    adata.obs['gene_name'] = y.index
    adata.var['gene_name'] = adata.var.index

    all_genes = adata.var['gene_name']

    save_path = Path('tmp/dspin_save_path')
    save_path.mkdir(exist_ok=True, parents=True)
    num_spin = len(adata.var)
    model = DSPIN(adata, str(save_path), num_spin=num_spin)

    # Data is filtered in the model creation, so extract the updated
    # data to continue with
    adata = model.adata
    num_spin = model.num_spin

    # # Custom prior h for perturbations
    # # dspin bioarxiv at least used to describe this as the way
    # perturbation_list = np.unique(adata.obs['gene_name'])
    # rows = np.array(adata.var.gene_name)
    # cols = perturbation_list
    # cur_h = np.zeros((len(rows), len(cols)))
    # for ii in range(len(rows)):
    #     for jj in range(len(cols)):
    #         if rows[ii] == cols[jj]:
    #             cur_h[ii, jj] = -1.5
    # extra_params = {'cur_h': cur_h}

    extra_params = {}


    # p = np.zeros((len(rows), len(cols)))
    # for ii in range(len(rows)):
    #     for jj in range(len(cols)):
    #         if rows[ii] == cols[jj]:
    #             p[ii, jj] = -1
    #

    params={'stepsz': 0.05, 'lambda_l1_j': 0.01, 'lambda_l2_h': 3}
    all_params = {**params, **extra_params}

    anton_util.log_timestamp(f'{all_params = }')

    # p = data['P']
    # p = p.loc[:, adata.var['gene_name']]
    # p = p.to_numpy()

    model = DSPIN(adata, str(save_path), num_spin=num_spin)
    model.network_inference(
        sample_id_key = 'gene_name',
        method = 'pseudo_likelihood',
        params = all_params,
        directed = True,
        # perturb_matrix = p,
        )

    estimated_network = pd.DataFrame(
            data = model.network,
            index = adata.var.gene_name,
            columns = adata.var.gene_name,
            )

    # Using these conversions to put the previously filtered out genes in again
    estimated_network = gs.util.edgelist_to_matrix(
        np.array(gs.util.matrix_to_edgelist(estimated_network)),
        all_genes = all_genes,
        )

    return estimated_network




def genie3_inference(data):

    import copy
    expression_data = copy.deepcopy(data['Y'])

    # Genie3 normalises the the output gene in each RF regression by
    # it's standard deviation. Some are 0 and crashes.
    # Took a note of looking into why they are 0
    # Filtering those out in the meantime
    all_genes = expression_data.columns
    stds = expression_data.std(axis = 0)
    non_zero_stds = stds > 0
    expression_data = expression_data.loc[:, non_zero_stds]

    # # Debug
    # expression_data = expression_data.iloc[:, :5]

    from GENIE3.GENIE3_python.GENIE3 import GENIE3  # pyright: ignore
    VIM = GENIE3(np.array(expression_data))
    estimated_network = pd.DataFrame(
        data = VIM,
        index = expression_data.columns,
        columns = expression_data.columns,
        )
    # genie3 does output on format regulators x targets
    # https://github.com/vahuynh/GENIE3/blob/master/GENIE3_python/GENIE3_python_doc.pdf
    # , but the group experience is that it often gets
    # the directionality inverted
    # It seems to do that with genesnake data as well, so
    # helping out by transposing.
    estimated_network = estimated_network.T

    # Using these conversions to put the previously filtered out genes in again
    estimated_network = gs.util.edgelist_to_matrix(
        np.array(gs.util.matrix_to_edgelist(estimated_network)),
        all_genes = all_genes,
        )

    return estimated_network


def deepsem_inference(data):
    import argparse
    import tempfile
    import os
    import shutil
    import sys
    from pathlib import Path

    import copy
    expression_data = copy.deepcopy(data['Y'])

    # # Debug
    # expression_data = expression_data.iloc[:, :10]

    all_genes = expression_data.columns
    stds = expression_data.std(axis = 0)
    non_zero_stds = stds > 0
    expression_data = expression_data.loc[:, non_zero_stds]

    # Save the original working directory so we can restore it later
    original_dir = os.getcwd()
    temp_file = None
    temp_dir = None

    try:
        # Change to DeepSEM directory and add to path
        deepsem_dir = Path('DeepSEM').resolve()
        os.chdir(deepsem_dir)
        sys.path.insert(0, str(deepsem_dir))

        # Import DeepSEM test model (for inference without ground truth)
        from src.DeepSEM_cell_type_test_non_specific_GRN_model import test_non_celltype_GRN_model

        # Create opt object with test parameters (for inference without ground truth)
        opt = argparse.Namespace(
            task='non_celltype_GRN',
            setting='test',
            n_epochs=120,
            # debug
            # n_epochs=20,
            batch_size=64,
            alpha=100,
            beta=1,
            lr=1e-4,
            lr_step_size=0.99,
            gamma=0.95,
            n_hidden=128,
            K=1,
            K1=1,
            K2=2,
            net_file=None,
        )

        # Create temporary file for data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name
            expression_data.to_csv(f.name)
            opt.data_file = f.name

        # Create temporary directory for save_name
        temp_dir = tempfile.mkdtemp()
        opt.save_name = temp_dir

        model = test_non_celltype_GRN_model(opt)
        model.train_model()

        # Read the output - based on collaborator code, it creates "GRN_inference_result.tsv"
        result_file = os.path.join(temp_dir, "GRN_inference_result.tsv")
        if os.path.exists(result_file):
            result_df = pd.read_csv(result_file, sep='\t')
        else:
            raise RuntimeError("DeepSEM did not produce expected output file")

    finally:
        # Restore the original working directory
        os.chdir(original_dir)
        # Cleanup temp file and temp dir
        if temp_file is not None and os.path.exists(temp_file):
            os.unlink(temp_file)
        if temp_dir is not None and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    estimated_network = result_df

    # Using these conversions to put the previously filtered out genes in again
    estimated_network = gs.util.edgelist_to_matrix(
        np.array(estimated_network),
        all_genes = all_genes,
        )

    return estimated_network






def psgrn_inference(data):

    from psgrn_extract_src.main import Custom as Model

    import copy
    expression_data = copy.deepcopy(data['Y'])
    # The method normalises by standard deviation. Some are 0 and crashes.
    # Took a note of looking into why they are 0
    # Filtering those out in the meantime
    all_genes = expression_data.columns
    stds = expression_data.std(axis = 0)
    non_zero_stds = stds > 0
    expression_data = expression_data.loc[:, non_zero_stds]

    Y = expression_data
    # Follows PSGRN code variable naming below.
    # They optionally do training and test set splits, hence training.
    expression_matrix_train = np.array(Y)
    interventions_train = np.array(Y.index)
    # The PSGRN model code expects this name for 
    # control cells that were not perturbed
    interventions_train[interventions_train == 'control'] = 'non-targeting'
    gene_names = list(Y.columns)

    model = Model()
    _, edgelist = model(
        expression_matrix = expression_matrix_train,
        interventions = list(interventions_train),
        gene_names = gene_names,
        # This training regime argument is not actually used by the model code
        # Giving it a None, because the argument does not have a default
        training_regime = None, # type: ignore[reportArgumentType]
        # Defaults to 0
        # self.model_seed,
    )

    # Using these conversions to put the previously filtered out genes in again
    estimated_network = gs.util.edgelist_to_matrix(
        np.array(edgelist),
        all_genes = all_genes,
        )
    print(estimated_network)

    return estimated_network










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



def bin_bulk(
        mat,
        # matrices,
        n_pseudo_bulks = 5,
        verbose = False,
        ):
    import pandas as pd
    import math

    # Based on initial testing, pseudo bulking is a
    # significant part of the runtime for the lsco method

    smallest_intended_bin = 5
    n_requested_pseudo_bulks = n_pseudo_bulks

    max_complaints = 3
    complaints_count = 0
    complain = True


    mat_bulk = []
    mat_indices = []
    # pseudo_bulk_indices = []



    groups = np.unique(mat.index)
    for group in groups:
        iis = np.nonzero(mat.index == group)[0]
        # cells = P.iloc[iis, :]
        n_cells = iis.shape[0]





    # for perturbed_gene in P:
        # If it wasn't for counting complaints, I would put all
        # the next paragraph away in a neat and tidy function.
        # Can do workarounds like having part of the logic here
        # and part in a function, but didn't find a way that felt good.
        # So just inlined all for now
        if (
                complaints_count >= max_complaints and
                not verbose and
                # This needed to avoid repeatedly entering here and priting
                # that it'll stop complaining
                complain
                ):
            print()
            print(f'Reached warning count limit of {max_complaints}, so will stop warning. To print all warnings, call with verbose = True')
            print()
            complain = False

        # perturbed_cell_indices = np.nonzero(P[perturbed_gene])[0]
        # n_cells = len(perturbed_cell_indices)

        # if n_cells == 0:
        #     complaints_count = complaints_count + 1
        #     print(f'Gene {perturbed_gene} has no perturbations, skipping.')
        #     continue

        n_pseudo_bulks = math.floor(n_cells / smallest_intended_bin)
        if n_pseudo_bulks > n_requested_pseudo_bulks:
            n_pseudo_bulks = n_requested_pseudo_bulks
        elif n_pseudo_bulks < n_requested_pseudo_bulks:
            complaints_count = complaints_count + 1
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
                print('remaining cells go with the last pseudo bulk')




        # # ..indices[0] is arbitrary
        # # The code assumes single perturbations, so all rows pointed to by the
        # # indices should be identical
        # #
        # # Actually worth checking!
        # # Not unthinkable that this or some other dataset will include
        # # multi target perturbations, and if so we want to know
        # #
        # # Actually, the way P is extracted, that shouldn't happpen.
        # # So such checks would need to be before extracting P
        # #
        # # Can comment this out to save time, if it turns out to be significant
        # first = P.iloc[perturbed_cell_indices[0], :]
        # for ii in perturbed_cell_indices:
        #     if not (P.iloc[ii, :] == first).all():
        #         raise ValueError(f'Perturbed cell indices for gene {perturbed_gene} do not point to identical rows in P.')
        # #
        # for _ in range(n_pseudo_bulks):
        #     mat_bulk.append(list(first))



        # If this function keeps just dealing with one matrix at a time,
        # I could rewrite this function to just chunk the matrix directly.
        # But for now, I have a function that chunks an array (of indices),
        # so I'll keep using that for now. It does the thing
        indices_chunks = chunk_array(iis, n_pseudo_bulks)

        for iiis in indices_chunks:
            mat_indices.append(group)
            mat_bulk.append(mat.iloc[iiis, :].mean(axis = 0))



        # pseudo_bulk_indices.append(indices_chunks)

    P_bulk_df = pd.DataFrame(mat_bulk)
    P_bulk_df.columns = mat.columns
    P_bulk_df.index = mat_indices

    # pseudo_bulks = {}
    # for matrix_name, matrix in matrices.items():
    #     matbulk = []
    #     for gene_pseudo_bulks in pseudo_bulk_indices:
    #         for index_array in gene_pseudo_bulks:
    #             matbulk.append(list(matrix.iloc[index_array, :].mean(axis = 0)))
    #     df = pd.DataFrame(matbulk)
    #     df.columns = matrix.columns
    #     pseudo_bulks[matrix_name] = df

    return P_bulk_df






def calculate_zero_fraction(df):
    non_zeros = np.nonzero(df)[0].shape[0]
    d = 1 - non_zeros / df.size
    return d






def transform(df, transform):
    # Note that this uses the natural logarithm.
    # Numpy has a readymade for it. Better calculation precision
    # than log2 it seemed when I looked at docs
    if transform == 'log1p':
        return np.log1p(df)
    elif transform == 'zscores':
        return (df - df.mean()) / df.std()
    # Does nothing useful, but enables fewer special cases in the pipeline code
    elif transform == 'raw':
        return df
    else:
        raise ValueError(f'Transform {transform} not recognised')







def merge_p_into_y(Y, P):
    index = np.array(Y.index).astype(str)
    for gene in P:
        perturbed_cell_indices = np.nonzero(P[gene])[0]
        index[perturbed_cell_indices] = gene
    Y.index = index
    return Y
