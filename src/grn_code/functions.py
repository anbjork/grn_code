
import numpy as np
import pandas as pd
import genesnake as gs



def Y_from_adata(adata):
    Y = adata.to_df()
    Y.index = adata.obs['gene']
    Y.index.name = 'perturbations'
    Y.columns = adata.var['gene_name']
    Y.columns.name = 'genes'
    return Y




def shuffle_y(dataset, shuffle_bool, **kwargs):
    import random
    import copy
    
    y = dataset['Y']
    if shuffle_bool:
        l = y.shape[0]
        riis = random.sample(population=range(l), k=l)
        y = copy.deepcopy(y)
        y = pd.DataFrame(
            data = np.array(y)[riis, :],
            index = y.index,
            columns = y.columns,
            )
    out = update_dataset_default(dataset, y, 'shuffle', shuffle_bool)

    return out





def normalise(dataset, do_or_not: bool, **kwargs):
    import numpy as np

    Y = dataset['Y']
    meta_data_label = kwargs['meta_data_label']

    if not do_or_not:
        out = update_dataset_default(dataset, Y, meta_data_label, do_or_not)
        return out

    # # Some diagnostics ran on some genespider simulated data
    # print(np.median(np.sum(np.array(dataset['Y']), axis = 1)))
    # # Somewhere around e3, so picking e3 as the normalisation target

    array = np.array(Y)
    cell_sums = np.sum(array, axis = 1)
    Y = pd.DataFrame(
        data = 1e3 * array / cell_sums[:, np.newaxis],
        index = Y.index,
        columns = Y.columns,
        )

    # # Check that it worked
    # print(Y.sum(axis = 1))

    out = update_dataset_default(dataset, Y, meta_data_label, do_or_not)
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

    # # Make benchmarking undirected
    # # Temporary. Horribly slow. Will fix if I want to keep running this
    # for network in (harmonised_estimated_network, harmonised_reference_network):
    #     for ii in range(network.shape[0]):
    #         for jj in range(network.shape[1]):
    #             m = max(network.iloc[ii, jj], network.iloc[jj, ii])
    #             network.iloc[ii, jj] = m
    #             network.iloc[jj, ii] = m

    tmp = gs.benchmarking.benchmark(
        estimated_network=harmonised_estimated_network,
        reference_network=harmonised_reference_network.astype(bool),
        method_name=method,
        fix_pr_ylim = True,
        return_plots = False,
        )
    # Does this reduce space use? For a relatively large run,
    # it saved about 2 GB of disk. The preprocessed data for that run
    # takes 7.6 GB, so no important difference. Also didn't notice
    # a meaningful speed difference, based on time stamps in logs,
    # for some partial test runs. Leaving the plots in by default.
    # [tmp.pop(s) for s in ['plot_roc', 'plot_pr']]
    #
    # Above outdated, because added option to return the data rather
    # than the plots themselves
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
        dataset,
        n_pseudo_bulks = 5,
        verbose = False,
        terse = False,
        **kwargs,
        ):
    # Based on initial testing, pseudo bulking is a
    # significant part of the runtime for the lsco method

    import pandas as pd
    import math

    # Needed to comply with calling interface, that allows non modifications
    if n_pseudo_bulks == False:
        out = update_dataset_default(dataset, dataset['Y'], 'pseudo_bulk', False)
        return out

    mat = dataset['Y']

    smallest_intended_bin = 5
    n_requested_pseudo_bulks = n_pseudo_bulks

    max_complaints = 3
    complaints_count = 0
    complain = True
    if terse:
        complain = False

    mat_bulk = []
    mat_indices = []

    groups = np.unique(mat.index)
    for group in groups:
        iis = np.nonzero(mat.index == group)[0]
        n_cells = iis.shape[0]
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

        n_pseudo_bulks = math.floor(n_cells / smallest_intended_bin)
        if n_pseudo_bulks > n_requested_pseudo_bulks:
            n_pseudo_bulks = n_requested_pseudo_bulks
        elif n_pseudo_bulks < n_requested_pseudo_bulks:
            complaints_count = complaints_count + 1
            if terse:
                print('some pseudo bulks have few cells')
                terse = False
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

        # This code structure is a left over from an earlier version,
        # where this functions handled both Y and P matrices.
        # Now it only needs to deal with the Y matrix, since P is extracted
        # from Y at a later stage of pre processing.
        # If this function keeps just dealing with one matrix at a time,
        # I could rewrite this function to just chunk the matrix directly.
        # But for now, I have a function that chunks an array (of indices),
        # so I'll keep using that for now. It does the thing
        indices_chunks = chunk_array(iis, n_pseudo_bulks)
        for iiis in indices_chunks:
            mat_indices.append(group)
            mat_bulk.append(mat.iloc[iiis, :].sum(axis = 0))

    pseudo_bulk_df = pd.DataFrame(mat_bulk)
    pseudo_bulk_df.columns = mat.columns
    pseudo_bulk_df.index = mat_indices

    out = update_dataset_default(
            dataset, pseudo_bulk_df, 'pseudo_bulk', n_pseudo_bulks)
    return out







def calculate_zero_fraction(df):
    non_zeros = np.nonzero(df)[0].shape[0]
    d = 1 - non_zeros / df.size
    return d



def update_dataset_default(dataset, new_Y, meta_field = None, label = None):
    # Note that this function updates dataset in place.
    # Caller has to handle copying if needed
    if meta_field is not None:
        dataset['meta'][meta_field] = label
    dataset['Y'] = new_Y
    return dataset



def transform(dataset, transform, **kwargs):
    df = dataset['Y']
    # Note that this uses the natural logarithm.
    # Numpy has a readymade for it. Better calculation precision
    # than log2 it seemed when I looked at docs
    if transform == 'log1p':
        df = np.log1p(df)
    elif transform == 'zscores':
        df = (df - df.mean()) / df.std()
    elif transform == 'none':
        pass
    else:
        raise ValueError(f'Transform {transform} not recognised')
    updated_dataset = update_dataset_default(
            dataset, df, kwargs['meta_data_label'], transform)
    return updated_dataset





def scanpy_preprocess(dataset, option, **kwargs):
    import scanpy as sc
    _ = option
    Y = dataset['Y']
    obs = pd.DataFrame({'perturbation': list(Y.index)})
    obs.index = obs.index.astype(str)
    var = pd.DataFrame({'gene_name': list(Y.columns)})
    var.index = var.index.astype(str)
    adata = sc.AnnData(X = Y.values, obs = obs, var = var)

    # Diagnostics. Commented out for performance when not actually looked at
    # sc.pp.calculate_qc_metrics(
    #         adata,
    #         inplace=True,
    #         # Need to disable this, because data dimensions doesn't match
    #         # its default assumption.
    #         # Not using the metric calculated with this argument anyway
    #         percent_top = None,
    #         )
    # sc.pl.violin(
    #         adata, ['n_genes_by_counts', 'total_counts'],
    #         jitter=0.4, multi_panel=True, 
    #         save=f"_dataset_{kwargs['index']}_qc_violin.png")
    # sc.pl.scatter(
    #         adata, x='total_counts', y='n_genes_by_counts',
    #         save=f"_dataset_{kwargs['index']}_counts_vs_genes.png"
    #         )
    # print()
    # print(kwargs['index'])
    # print(dataset['meta']['dataset_parameters']['data_case'])
    # print(adata.shape)
    # Remove cells with too few genes detected

    nz = (Y > 0)
    genes_per_cell = np.sum(nz, axis = 1)
    genes_cutoff = np.quantile(genes_per_cell, 0.1)

    cell_counts = np.sum(Y, axis = 1)
    counts_cutoff = np.quantile(cell_counts, 0.1)

    sc.pp.filter_cells(adata, min_genes = genes_cutoff)
    sc.pp.filter_cells(adata, min_counts = counts_cutoff)

    sc.pp.highly_variable_genes(adata, n_top_genes=100, flavor='seurat_v3')
    # Commented out for performance when not actually looked at
    # sc.pl.highly_variable_genes(
    #         adata,
    #         save = f'_dataset_{kwargs["index"]}_highly_variable_genes.png'
    #         )
    adata = adata[:, adata.var['highly_variable']]  # pyright: ignore
    # Slicing makes view, but then subsequent sc.pp functions cannot operate
    # on view, so they copy. However, they also incessantly warn that they
    # are copying. Very annoying. In practice, they defferred the copying
    # from the slicing to the next thing, which is rarely a practically
    # useful optimisation. This at the cost of spamming users with
    # warnings, even when they are executing a supported happy case.
    # Very strange design.. Anyhow, already spent a bunch of time
    # understanding how and why it's broken like this, so moving on
    adata = adata.copy()

    # Specifically remove cells that have 0 HVG counts.
    # Could remove ones with low counts too, but that's not 
    # biologically and statistically sound on a HVG subset, and 
    # might theoretically also reduce some genes to 0 counts, making the
    # data column rank deficient.
    # The reason for cleaning up empty cells is to avoid NaN values in
    # subsequent normalisation
    sc.pp.filter_cells(adata, min_counts = 1)

    df = pd.DataFrame(
            adata.X,
            index = adata.obs.perturbation,
            columns = adata.var.gene_name
            )
    updated_dataset = update_dataset_default(dataset, df)
    return updated_dataset




def record_dropout_fractions(datasets, when):
    for d in datasets:
        Y = d['Y']
        all = Y
        controls = Y.loc[Y.index == 'control', :]
        perturbed = Y.loc[~(Y.index == 'control'), :]
        for subset, name in zip(
                    [all, controls, perturbed], 
                    ['all', 'controls', 'perturbed']
                    ):
            d['meta'][f'0_fraction__{when}__{name}'] = (
                    calculate_zero_fraction(subset))





def compute_differences(dataset, do_or_not: bool, **kwargs):

    Y = dataset['Y']
    if not do_or_not:
        out = update_dataset_default(dataset, dataset['Y'], 'control_delta', False)
        return out

    ct_bool = (Y.index == 'control')
    control_cells = Y.loc[ct_bool, :]
    control = control_cells.mean(axis=0)
    Y = Y.loc[~ct_bool, :]
    delta = pd.DataFrame(
        Y - control,
        index = Y.index,
        columns = Y.columns
    )

    out = update_dataset_default(dataset, delta, 'control_delta', True)
    return out








def merge_p_into_y(Y, P):
    index = np.array(Y.index).astype(str)
    for gene in P:
        perturbed_cell_indices = np.nonzero(P[gene])[0]
        index[perturbed_cell_indices] = gene
    Y.index = index
    return Y



