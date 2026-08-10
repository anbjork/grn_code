
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

    return {'correlation': estimated_network}


def random_inference(data):

    Y = data['Y']

    rng = np.random.default_rng()
    random_array = rng.random([len(Y.columns)] * 2)

    estimated_network = pd.DataFrame(
        data = random_array,
        index = Y.columns,
        columns = Y.columns
    )

    return {'random': estimated_network}


def fast_methods_inference(data):

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
    try:
        en = gs.inference.infer_networks(
            Y=Y, P=P,
            method=m)
        en[np.isnan(en)] = 0
        estimated_networks[m] = en
    except Exception as e:
        print(f'{m} failed with:')
        print(e)


    return estimated_networks





def dspin_inference_wrapper(data):
    from pprint import pprint

    # options = {
    #     'use_perturbation_matrix': [True, False],
    #     'perturbation_knockdown_value': [-1, -0.9, -0.7],
    #     'use_prior_h': [True, False],
    #     }
    # configs = []
    # def recursive_setting(settings_applied, settings_left):
    #     from copy import deepcopy
    #     settings_applied = deepcopy(settings_applied)
    #     settings_left = deepcopy(settings_left)
    #     if len(settings_left) == 0:
    #         configs.append(deepcopy(settings_applied))
    #         return
    #     new_setting, options = settings_left.popitem()
    #     for option in options:
    #         settings_applied[new_setting] = option
    #         recursive_setting(
    #             settings_applied = settings_applied,
    #             settings_left = settings_left
    #             )
    # recursive_setting({}, options)
    # pprint(configs)

    # Manual override
    configs = [{
        'use_perturbation_matrix': True,
        'perturbation_knockdown_value': -0.9,
        'use_prior_h': True,
        }]

    estimated_networks = {}
    for config in configs:
        anton_util.log_timestamp(f'Running dspin...')
        print('config:')
        pprint(config)
        en = dspin_inference(data=data, **config)
        # It's always length 1
        ((method_name, estimated_network),) = en.items()
        config_cat = '__'.join([f'{k}_{v}' for k, v in config.items()])
        method_name = f'{method_name}__{config_cat}'
        estimated_networks[method_name] = estimated_network

    return estimated_networks


def dspin_inference(
        data,
        use_perturbation_matrix = None,
        perturbation_knockdown_value = -1,
        use_prior_h = None,
        ):
    import anndata as ad
    from dspin import DSPIN
    from copy import deepcopy

    # Recreating an anndata object to fit with the rest of the code
    # from previously.
    # y is a regular pandas dataframe of expression values,
    # where the index/rownames is the gene perturbed or 'control',
    # and the columns are the genes
    y = data['Y']

    # # Debug
    # y = y.iloc[:, :10]

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
    additional_params = {
            # 'discretize_params' : {
            #     'clip_percentile': 100,
            #     },
            'filter_threshold' : 0.06,
            }
    model = DSPIN(
            deepcopy(adata), str(save_path), num_spin=num_spin,
            **additional_params,  # pyright: ignore
            )

    # # Data is filtered in the model creation, so extract the updated
    # # data to continue with
    # adata = model.adata
    # num_spin = model.num_spin

    # In case dspins internal preprocessing applied something that is not
    # idempotent, let's actually filter the original data, and go from
    # that again.
    #
    # Also, try filtering out cells where a gene was perturbed that is not
    # among the genes in the data (columns).
    #
    # Didn't seem to improve results.
    genes_left = model.adata.var.gene_name
    print(f'{genes_left = }')
    to_keep = list(genes_left) + ['control']
    # Since genes are not in adata.obs_names
    to_keep_bool = adata.obs['gene_name'].isin(to_keep)
    adata = adata[to_keep_bool, genes_left]  # pyright: ignore
    num_spin = len(adata.var)
    print(adata)

    # perturbation_list: Created similarly by getting unique elements from
    # a full list of samples in raw_data_state -> sample_states
    # So hopefully the same. But it seems sensitive to sorting, so if 
    # sorted somewhere else, it might lead to problems.
    perturbation_list = np.unique(adata.obs['gene_name'])
    rows = np.array(adata.var.gene_name)
    cols = perturbation_list

    extra_params = {}

    if use_prior_h:
        # Custom prior h for perturbations
        # dspin bioarxiv describes this as the way
        cur_h = np.zeros((len(rows), len(cols)))
        for ii in range(len(rows)):
            for jj in range(len(cols)):
                if rows[ii] == cols[jj]:
                    cur_h[ii, jj] = -1.5
        extra_params = {'cur_h': cur_h}
        # My intuition would have been the other transpose, ie
        # samples x genes, but in learn_network_adam, it gets dimensions as
        # num_spin, num_round = train_dat['cur_h'].shape
        # , which implies genes x samples, since num_spin == number of variables,
        # so maybe those are the internal dimensions
        #
        # Seeing the printout of the cur_h matrix, I remember that it's not
        # as simple as genes x samples, the samples dimension is substituted
        # by a smaller dimension which I am a bit unclear on
        #
        # In sample_states, the state_list assigned to _raw_data is created.
        # _raw_data determines the dimensions of cur_h in default_params. 
        # It comes from a unique list of samples. 
        # So h is genes x unique(samples), which
        # matches above. I guess it's sorting sensitive, since just applying
        # unique to the sample lables from the data. That's one thing that
        # can go wrong I guess
        #
        # Crashes when run with transpose, so at least dimensions are probably right.


    if use_perturbation_matrix:
        p = np.zeros((len(rows), len(cols)))
        for ii in range(len(rows)):
            for jj in range(len(cols)):
                if rows[ii] == cols[jj]:
                    p[ii, jj] = perturbation_knockdown_value
        # Looks like wrong transpose here, based on github readme.
        # But in apply_regularization, it is subtracted from h_rela,
        # which probably is the same dimensions as cur_h, so might be
        # a mistake in the readme. Worth trying both orientations.
        #
        # Crashes with transpose, actually exactly on the line subtracting 
        # from h_rela. So these dimensions seem right.

        # Alternative construction of the perturbation matrix
        # Should be equivalent
        # p = data['P']
        # p = p.loc[:, adata.var['gene_name']]
        # p = p.to_numpy()
        # No, not equivalent.  Note that perturbation list is
        # a list of unique perturbations, so not the same dimensions.
    else:
        # Checked network_inference, and perturb_matrix does not go in params,
        # but the case perturb_matrix == None is handled
        p = None


    params={'stepsz': 0.05, 'lambda_l1_j': 0.01, 'lambda_l2_h': 3}
    all_params = {**params, **extra_params}

    anton_util.log_timestamp(f'{all_params = }')

    model = DSPIN(
            adata, str(save_path), num_spin=num_spin,
            **additional_params,  # pyright: ignore
            )
    model.network_inference(
        sample_id_key = 'gene_name',
        method = 'pseudo_likelihood',
        params = all_params,
        directed = True,
        perturb_matrix = p,  # pyright: ignore
        )

    # Note: Assumes orientation regulators x targets
    # Quick test using other transpose did not seem to improve results
    estimated_network = pd.DataFrame(
            data = model.network,
            index = adata.var.gene_name,
            columns = adata.var.gene_name,
            )
    # estimated_network = estimated_network.T

    # Using these conversions to put the previously filtered out genes in again
    edgelist = gs.util.matrix_to_edgelist(estimated_network)
    estimated_network = gs.util.edgelist_to_matrix(
        regulators = edgelist['regulator'],
        targets = edgelist['target'],
        values = edgelist['value'],
        all_genes = all_genes,
        )

    return {'dspin': estimated_network}




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
    edgelist = gs.util.matrix_to_edgelist(estimated_network)
    estimated_network = gs.util.edgelist_to_matrix(
        regulators = edgelist['regulator'],
        targets = edgelist['target'],
        values = edgelist['value'],
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

    # NOTE:
    # changed the edgelist_to_matrix function interface.
    # deepsem needs a gpu, so cannot be tested on octa.
    # adjusted the code, but haven't tested it since.
    # so if it crashes, look at the format of estimated_network
    # //AB
    estimated_network = gs.util.edgelist_to_matrix(
        regulators = estimated_network.iloc[:, 0],
        targets = estimated_network.iloc[:, 1],
        values = estimated_network.iloc[:, 2],
        all_genes = all_genes,
        )

    return {'deepsem': estimated_network}






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




    with np.errstate(divide='ignore', invalid='ignore'):
		# psgrn spams warnings
		# 	messages
		# 		/home/anbjork/glob_venv/lib/python3.10/site-packages/numpy/lib/_function_base_impl.py:3046: RuntimeWarning: invalid value encountered in divide
		# 		  c /= stddev[None, :]
		# 	debugged
		# 	it picks subsets of observations to calculate correlations
		# 	and some of those subsets are all 0s
		# 	which gives nan for that correlation
		# 	probably happens more without pseudo bulking
		# 	that may be why I encounter it now
		# 	tried with a pseudo bulked dataset
		# 	no warnings
		# 	okay, cool
		# 	then I know where it comes from
		# 	I'll try silencing the warning
		# 	See if it works
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
        regulators = edgelist['regulator'],
        targets = edgelist['target'],
        values = edgelist['value'],
        all_genes = all_genes,
        )
    print(estimated_network)

    return {'psgrn': estimated_network}


def inspre_inference(data):
    import subprocess
    import tempfile
    import os
    import json
    import copy
    from pathlib import Path
    import anndata as ad

    expression_data = copy.deepcopy(data['Y'])
    all_genes = expression_data.columns

    # Filter out zero-std genes since methods crashes with those
    stds = expression_data.std(axis=0)
    non_zero_stds = stds > 0
    expression_data = expression_data.loc[:, non_zero_stds]

    # Get temporary file paths
    with tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False) as f:
        input_path = f.name
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        output_path = f.name
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        config_path = f.name

    anton_util.log_timestamp('Preparing AnnData for INSPRE...')
    # Convert 'control' to 'non-targeting' as expected by INSPRE
    perturbations = expression_data.index.to_series()
    # Seems the interface fit_inspre_from_X does use 'control',
    # while the fit_inspre_from_h5X interface uses 'non-targeting'
    # perturbations = perturbations.replace('control', 'non-targeting')
    var_df = pd.DataFrame(index=expression_data.columns)
    var_df['gene_name'] = expression_data.columns

    adata = ad.AnnData(
        X=expression_data.values,
        var=var_df
    )
    adata.write_h5ad(Path(input_path))

    # Create targets vector - one entry per cell indicating which gene was perturbed
    targets_vector = perturbations.tolist()

    config = {
        'input': input_path,
        'output': output_path,
        'targets': targets_vector,
        'ncores': 5,
        'weighted': True,
        'dag': False,
        'nlambda': 20,
        'iterations': 100,
        'verbose': 1,
        'max_med_ratio': 50.0,
        'cv_folds': 5
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Run R script
    anton_util.log_timestamp('Running INSPRE R script...')
    r_script_path = Path('inspre_integration_docs/run_inspre_benchmark.R').resolve()

    result = subprocess.run([
        'Rscript', str(r_script_path), config_path
    ], capture_output=True, text=True)

    if result.stdout:
        print("R stdout:", result.stdout)
    if result.stderr:
        print("R stderr:", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"INSPRE R script failed with return code {result.returncode}.\n"
            f"See R stderr above for details."
        )

    # Read results from JSON
    anton_util.log_timestamp('Reading INSPRE results...')
    with open(output_path, 'r') as f:
        results = json.load(f)

    # Diagnostic: print the top-level keys and shapes of what came back from R
    print("INSPRE result keys:", list(results.keys()))
    for k, v in results.items():
        if isinstance(v, list):
            if len(v) > 0 and isinstance(v[0], list):
                if len(v[0]) > 0 and isinstance(v[0][0], list):
                    print(f"  {k}: 3D list, dims={len(v)} x {len(v[0])} x {len(v[0][0])}")
                else:
                    print(f"  {k}: nested list (matrix), outer len={len(v)}, inner len={len(v[0])}")
            else:
                print(f"  {k}: flat list, len={len(v)}, first few={v[:3]}")
        else:
            print(f"  {k}: {type(v).__name__} = {v}")
    # Print G_hat as numpy array shape
    if 'G_hat' in results:
        G_hat_np = np.array(results['G_hat'])
        print(f"  G_hat as np.array shape: {G_hat_np.shape}")

    # G_hat shape is (nlambda, D, D); use the CV-selected best lambda index from R
    G_hat_np = np.array(results['G_hat'])  # shape: (nlambda, D, D)
    result_genes = results['G_hat_rownames']
    best_lambda_idx = results['best_lambda_idx'] - 1  # R is 1-indexed, Python is 0-indexed
    network_matrix = G_hat_np[best_lambda_idx]

    # Diagnostic: print G_hat at best lambda (Python side) for comparison with R
    print(f"\n--- G_hat at best lambda (Python, index {best_lambda_idx}) ---")
    print("Genes:", result_genes)
    print(pd.DataFrame(network_matrix, index=result_genes, columns=result_genes))
    print("---------------------------------------------------\n")
    network_matrix = np.array(network_matrix, dtype=object)
    network_matrix[network_matrix == 'NA'] = np.nan
    network_matrix = network_matrix.astype(float)
    estimated_network = pd.DataFrame(
        data=network_matrix,
        index=result_genes,
        columns=result_genes
    )
    estimated_network = estimated_network.fillna(0)

    # Using these conversions to put the previously filtered out genes in again
    edgelist = gs.util.matrix_to_edgelist(estimated_network)
    estimated_network = gs.util.edgelist_to_matrix(
        regulators=edgelist['regulator'],
        targets=edgelist['target'],
        values=edgelist['value'],
        all_genes=all_genes,
    )

    # Clean up temporary files
    for path in [input_path, output_path, config_path]:
        if os.path.exists(path):
            os.unlink(path)

    return {'inspre': estimated_network}


def inspre_inference_hdf5(data):
    """
    INSPRE inference using the fit_inspre_from_h5X interface, mirroring the
    gwps analysis pipeline. This includes guide effect calculation and filtering
    before fitting, as done in run_gwps_analysis.R.

    Key differences from inspre_inference:
    - Uses fit_inspre_from_h5X instead of fit_inspre_from_X
    - Data is written cells x genes to HDF5; HDF5's row/column-major conversion
      between Python (row-major) and R (column-major) means R reads it as
      genes x cells, which is what fit_inspre_from_h5X expects.
    - Controls are labelled 'non-targeting'
    - obs table includes 'perturbation' and 'guide_id' columns
    - var table includes 'gene_name' column
    """
    import subprocess
    import tempfile
    import os
    import json
    import copy
    from pathlib import Path
    import h5py

    expression_data = copy.deepcopy(data['Y'])
    all_genes = expression_data.columns

    # Filter out zero-std genes
    stds = expression_data.std(axis=0)
    non_zero_stds = stds > 0
    expression_data = expression_data.loc[:, non_zero_stds]

    # Map 'control' -> 'non-targeting' as expected by fit_inspre_from_h5X
    perturbations = expression_data.index.to_series().replace('control', 'non-targeting')

    # For the hdf5 interface, guide_id is used to match cells to targets.
    # Since our simulated data has one guide per perturbation target, we use
    # the perturbation name as the guide_id as well.
    guide_ids = perturbations.copy()

    gene_names = list(expression_data.columns)

    # Get temporary file paths
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        input_path = f.name
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        output_path = f.name
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        config_path = f.name

    anton_util.log_timestamp('Preparing HDF5 file for INSPRE (hdf5 interface)...')

    # Write HDF5 file in the format expected by fit_inspre_from_h5X.
    # We write X as cells x genes (the natural Python/NumPy row-major layout).
    # Because Python is row-major and R is column-major, hdf5r in R will read
    # this back as genes x cells — exactly the format fit_inspre_from_h5X expects.
    # No explicit transpose is needed here.
    with h5py.File(input_path, 'w') as hf:
        hf.create_dataset('X', data=expression_data.values)

        # obs group: per-cell metadata
        obs_grp = hf.create_group('obs')
        obs_grp.create_dataset(
            'perturbation',
            data=np.array(perturbations.tolist(), dtype=h5py.special_dtype(vlen=str))
        )
        obs_grp.create_dataset(
            'guide_id',
            data=np.array(guide_ids.tolist(), dtype=h5py.special_dtype(vlen=str))
        )

        # var group: per-gene metadata
        var_grp = hf.create_group('var')
        var_grp.create_dataset(
            'gene_name',
            data=np.array(gene_names, dtype=h5py.special_dtype(vlen=str))
        )

    config = {
        'input': input_path,
        'output': output_path,
        'ncores': 5,
        'dag': False,
        'nlambda': 20,
        'verbose': 1,
        'max_med_ratio': 50.0
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Run R script
    anton_util.log_timestamp('Running INSPRE R script (hdf5 interface)...')
    r_script_path = Path('inspre_integration_docs/run_inspre_benchmark_hdf5.R').resolve()

    result = subprocess.run([
        'Rscript', str(r_script_path), config_path
    ], capture_output=True, text=True)

    if result.stdout:
        print("R stdout:", result.stdout)
    if result.stderr:
        print("R stderr:", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"INSPRE hdf5 R script failed with return code {result.returncode}.\n"
            f"See R stderr above for details."
        )

    # Read results from JSON
    anton_util.log_timestamp('Reading INSPRE hdf5 results...')
    with open(output_path, 'r') as f:
        results = json.load(f)

    network_matrix = np.array(results['R_hat'])
    result_genes = results['R_hat_rownames']
    network_matrix = np.array(network_matrix, dtype=object)
    network_matrix[network_matrix == 'NA'] = np.nan
    network_matrix = network_matrix.astype(float)
    estimated_network = pd.DataFrame(
        data=network_matrix,
        index=result_genes,
        columns=result_genes
    )
    estimated_network = estimated_network.fillna(0)

    # Put previously filtered-out genes back in
    edgelist = gs.util.matrix_to_edgelist(estimated_network)
    estimated_network = gs.util.edgelist_to_matrix(
        regulators=edgelist['regulator'],
        targets=edgelist['target'],
        values=edgelist['value'],
        all_genes=all_genes,
    )

    # Clean up temporary files
    for path in [input_path, output_path, config_path]:
        if os.path.exists(path):
            os.unlink(path)

    return {'inspre_hdf5': estimated_network}



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
        dataset,
        n_pseudo_bulks = 5,
        verbose = False,
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



def update_dataset_default(dataset, new_Y, meta_field, label):
    from copy import deepcopy
    ds = deepcopy(dataset)
    ds['meta'][meta_field] = label
    ds['Y'] = new_Y
    return ds



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
