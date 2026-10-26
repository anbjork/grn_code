
import numpy as np
import pandas as pd
import genesnake as gs
from pathlib import Path
import anton_util



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








def zscore_max_variants(data):

    estimated_networks = {}

    P = data['P']
    Y = data['Y']

    m = 'zscore_ab'
    anton_util.log_timestamp(f'Running {m}...')
    en = gs.inference.infer_networks(
        Y=Y, P=P,
        method=m)
    en[np.isnan(en)] = 0
    for jj in range(en.shape[1]):
        for ii in range(en.shape[0]):
            a = en.iloc[ii, jj]
            b = en.iloc[jj, ii]
            minimum = np.argmin([a, b])
            if minimum == 0:
                en.iloc[ii, jj] = 0
            elif minimum == 1:
                en.iloc[jj, ii] = 0
    m = m + '_max'
    estimated_networks[m] = en

    # print(en)

    m = 'zscore_dream3'
    anton_util.log_timestamp(f'Running {m}...')
    en = gs.inference.infer_networks(
        Y=Y, P=P,
        method=m)
    en[np.isnan(en)] = 0
    for jj in range(en.shape[1]):
        for ii in range(en.shape[0]):
            a = en.iloc[ii, jj]
            b = en.iloc[jj, ii]
            minimum = np.argmin([a, b])
            if minimum == 0:
                en.iloc[ii, jj] = 0
            elif minimum == 1:
                en.iloc[jj, ii] = 0
    m = m + '_max'
    estimated_networks[m] = en



    return estimated_networks






def lasso_T_density(data, target_density):

    estimated_networks = {}

    P = data['P']
    Y = data['Y']

    m = 'lasso_density'
    anton_util.log_timestamp(f'Running {m}...')
    en = gs.inference.infer_networks(
        Y=Y, P=P,
        method=m,
        target_density = target_density)
    en = en.T

    m = m + f'_T_density_{target_density}'

    if np.any(np.isnan(en)):
        raise ValueError(f'NaN values in {m} output')

    estimated_networks[m] = en
    return estimated_networks





def zscore_without_controls(data):

    estimated_networks = {}

    P = data['P']
    Y = data['Y']

    Y = Y.loc[Y.index != 'control', :]
    P = P.loc[P.index != 'control', :]

    m = 'zscore_ab'
    anton_util.log_timestamp(f'Running {m}...')
    en = gs.inference.infer_networks(
        Y=Y, P=P,
        method=m)

    en[np.isinf(en)] = 0
    en[np.isnan(en)] = 0
    m = m + '_without_controls'
    estimated_networks[m] = en

    return estimated_networks





def lsco_T_without_controls(data):

    estimated_networks = {}

    P = data['P']
    Y = data['Y']

    Y = Y.loc[Y.index != 'control', :]
    P = P.loc[P.index != 'control', :]

    stds = Y.std(axis = 0)
    print('0 stds:')
    print(sum(stds == 0))
    Y = Y.loc[:, stds > 0]
    P = P.loc[:, stds > 0]

    m = 'lsco'
    m_name = f'{m}.T_without_controls_and_stange_columns'
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

    return estimated_networks







def perfect_inference(data, location_of_cheat_sheet):

    p = Path(location_of_cheat_sheet) / 'reference_networks.pkl'

    reference_networks = anton_util.unpickle_object(p)
    reference_networks_dict = {
            # frozensets are immutable, and so can be used as keys.
            # This way, it matches keys on all meta data fields, without
            # me needing to construct those keys manually.
            # Guarantees only matching if all meta data fits,
            # so no risk of errors
            # (Well, as long as meta data is complete, to uniquely identify)
            frozenset(r['meta']['dataset_parameters'].items()): r['data'] 
            for r in reference_networks
            }

    meta = data['meta']
    meta_key = frozenset(meta['dataset_parameters'].items())
    full = reference_networks_dict[meta_key]

    Y = data['Y']
    genes_in_data = Y.columns
    filtered = full.loc[genes_in_data, genes_in_data]

    return {
            'perfect_inference_all_genes': full,
            'perfect_inference_filtered_genes': filtered
            }




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

    if np.any(y < 0):
        raise ValueError('dspin is not constructed for negative values in Y')

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
                    cur_h[ii, jj] = -3
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

    # # Debug
    # expression_data = expression_data.iloc[:, :5]

    from grn_code.paths_anchor import repo_root
    import sys
    sys.path.insert(0, str(repo_root / 'GENIE3'))
    from GENIE3_python.GENIE3 import GENIE3  # pyright: ignore

    VIM = GENIE3(np.array(expression_data), nthreads = 5)
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

    return {'genie3.T': estimated_network}




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

    return {'deepsem': estimated_network}






def psgrn_inference(data):

    from grn_code.psgrn_extract_src.main import Custom as Model

    import copy
    expression_data = copy.deepcopy(data['Y'])

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

    estimated_network = gs.util.edgelist_to_matrix(
        regulators = edgelist['regulator'],
        targets = edgelist['target'],
        values = edgelist['value'],
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

    # Clean up temporary files
    for path in [input_path, output_path, config_path]:
        if os.path.exists(path):
            os.unlink(path)

    return {'inspre_hdf5': estimated_network}





def bigsm_inference(data):
    """
    BiGSM inference using MATLAB through an HDF5 interface (inspired by Anton)
    Python -> HDF5 -> MATLAB -> HDF5 -> Python

    MATLAB only runs bigsm.m with two local changes:
    The results in A are storted in row j also when the gene
    converges(typo fix, confirmed by Hang Qin),
    and genes with an all-zero row in P are skipped

    Input:
        data['Y'] : expression matrix, samples x genes
        data['P'] : perturbation matrix, samples x genes

    Output:
        'bigsm'   : the matrix as read back from the HDF5 file (A_matlab.T)
        'bigsm_T' : its transpose, i.e. bigsm.m's own matrix (A_matlab).
                    This is the variant that matches the reference networks.

    Orientation
    -----------
    HDF5 transposes 2D data between h5py (row-major) and MATLAB
    (column-major). Therfore:
      - Y and P, written as samples x genes, arrive in MATLAB as
        genes x samples, which is what bigsm.m expects.
      - run_bigsm_h5.m writes A as bigsm.m computes it, without transposing,
        so h5py reads it as A_matlab.T. That is 'bigsm'.
      - 'bigsm_T' = 'bigsm'.T = A_matlab.

    Why bigsm_T is the best variant
    -------------------------------
    In the exact, noise-free model, bigsm.m builds row j of A from the
    equationn of gene j only (P(j, :) is regressed on Y), i.e. it solves
    P = A Y with A[target, regulator]. One would therefore expect A_matlab to
    need a transpose to become [regulator, target], and that 'bigsm' (the
    array as read from the file) would be the one matching the reference
    networks. This is what the first version of this docstring assumed.

    The benchmark shows the opposite: on the simulated GeneSnake data
    bigsm_T matches the reference networks (AUROC about 0.85) and bigsm
    does not (about 0.55), in all dataset configurations. Checked:
      - No unexpected transpose in the HDF5 round trip, in
        run_bigsm_h5.m, in get_P / preprocessing or in the benchmark.
      - bigsm agrees with lsco as returned (rank correlation of absolute
        values about 0.98), not with lsco.T (about 0.04). So two different
        implementations, one in MATLAB and one in pure Python, give the same
        matrix, and bigsm_T lines up with lsco.T, which is also the variant
        that matches the reference. The effect is shared by the methods and
        not an artifact of the MATLAB / HDF5 route.
      - The reference networks are [regulator, target] in the sense of the
        data. R[j, k] = mean change of gene k in cells where gene j is
        knocked down (row = perturbed gene), computed directly from the
        data, matches the reference in position (AUROC about 0.85, against
        0.50 for the transposed reference).

    Likely reason (hypothesis, not verified): bigsm.m, like lsco, regresses
    the perturbation of gene j on the expression of all genes. In noisy
    single-cell data the best predictors of "gene j was knocked down" are
    the genes that respond to j, so row j of A_matlab carries the targets of
    j ([regulator, target], like the response matrix R) rather than the
    regulators of j, as the exact model would give.

    The orientation was chosen from the benchmark against the ground truth,
    so do not change it without rerunning the benchmark.

    Sign: bigsm.m fits P = +A Y, while GeneSPIDER models P = -A Y, so the
    weights have the opposite sign to lsco (signed correlation between bigsm
    and lsco about -0.91). This does not matter for AUROC on absolute
    values; take it into account if signed weights are used.
    """

    import subprocess
    import tempfile
    import os
    import copy

    import h5py
    import numpy as np
    import pandas as pd

    expression_data = copy.deepcopy(data['Y'])
    perturbations = data['P']

    # The paths are just strings inside a temporary directory. Nothing is
    # created up front: Python creates the input file below and MATLAB
    # creates the output file itself. The directory (and any files in it)
    # is removed automatically when the with-block ends, also on errors.
    with tempfile.TemporaryDirectory() as tmpdir:

        input_path = os.path.join(tmpdir, 'bigsm_input.h5')
        output_path = os.path.join(tmpdir, 'bigsm_output.h5')

        # write input
        anton_util.log_timestamp(
            'Preparing HDF5 file for BiGSM...'
        )

        print(
            'Python Y shape:',
            expression_data.shape
        )

        print(
            'Python P shape:',
            perturbations.shape
        )

        # sanity checks on the input before sending it to MATLAB
        for name, df in [('Y', expression_data), ('P', perturbations)]:
            values = df.values

            n_nan = int(np.isnan(values).sum())
            n_inf = int(np.isinf(values).sum())
            if n_nan or n_inf:
                raise ValueError(
                    f'{name} contains {n_nan} NaN and {n_inf} Inf values.'
                )

            constant_genes = np.where(values.std(axis=0) == 0)[0]
            if len(constant_genes) > 0:
                print(
                    f'Warning: {len(constant_genes)} constant genes '
                    f'in {name}: {list(df.columns[constant_genes])}'
                )

        with h5py.File(input_path, 'w') as hf:

            hf.create_dataset(
                'Y',
                data=expression_data.values
            )

            hf.create_dataset(
                'P',
                data=perturbations.values
            )

            hf.create_dataset(
                'max_iter',
                data=20
            )

        # run matlab
        anton_util.log_timestamp(
            'Running BiGSM MATLAB script...'
        )

        # The repo root comes from paths_anchor rather than from the location
        # of this script, which is more robust if the file is moved.
        from grn_code.paths_anchor import repo_root
        bigsm_dir = repo_root / 'BiGSM_ready'

        cmd = (
            f"addpath(genpath('{bigsm_dir}')); "
            f"run_bigsm_h5('{input_path}', '{output_path}');"
        )

        result = subprocess.run(
            [
                'matlab',
                '-batch',
                cmd
            ],
            capture_output=True,
            text=True
        )

        # print matlab output to make debugging easier
        if result.stdout:
            print('MATLAB stdout:')
            print(result.stdout)

        if result.stderr:
            print('MATLAB stderr:')
            print(result.stderr)

        if result.returncode != 0:
            raise RuntimeError(
                'BiGSM MATLAB script failed.\n'
                f'MATLAB return code: {result.returncode}'
            )

        # read result
        anton_util.log_timestamp(
            'Reading BiGSM results...'
        )

        with h5py.File(output_path, 'r') as hf:

            # h5py sees what MATLAB wrote as transposed: A = A_matlab.T
            # (see docstring)
            A = hf['A'][:]

        # sanity checks on A
        n_genes = len(
            expression_data.columns
        )

        expected_shape = (
            n_genes,
            n_genes
        )

        if A.shape != expected_shape:
            raise ValueError(
                f'Unexpected BiGSM network shape: {A.shape}. '
                f'Expected: {expected_shape}.'
            )

        if not np.isfinite(A).all():
            raise ValueError(
                'BiGSM returned NaN or Inf values.'
            )

        # 'bigsm' is the array as read from the HDF5 file (A_matlab.T).
        # 'bigsm_T' is its transpose, i.e. bigsm.m's own matrix (A_matlab),
        # and is the variant that matches the reference networks
        # (see docstring). Both are returned so the benchmark shows both.
        estimated_network = pd.DataFrame(
            data=A,
            index=expression_data.columns,
            columns=expression_data.columns
        )

        return {
            'bigsm': estimated_network,
            'bigsm_T': estimated_network.T
        }







def regdiffusion_inference(data):
   import copy
   import regdiffusion as rd

   expression_data = copy.deepcopy(data['Y'])

   # print("DEBUG shape:", expression_data.shape)
   # print("DEBUG describe:", expression_data.describe())
   # print("DEBUG min/max:", expression_data.min().min(), expression_data.max().max())
   X = expression_data.to_numpy()

   trainer = rd.RegDiffusionTrainer(X)
   trainer.train()

   adj = trainer.get_adj()

   estimated_network = pd.DataFrame(
       data=adj,
       index=expression_data.columns,
       columns=expression_data.columns,
   )

   return {'regdiffusion': estimated_network}






