


def configure():

    vars = {}

    old_dispersions = [0.1, 10, 15, 20, 50]
    additional_dispersions = [0.2, 0.3, 0.5, 1, 2, 3, 5, 7]
    even_more = [30, 40, 50, 70, 90]
    vars['dispersion'] = sorted(old_dispersions + additional_dispersions + even_more)

    snrs = [
            0.010, 0.015, 0.02,
            0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.07, 
            0.1, 0.2, 0.3, 0.5, 0.7, 
            1, 5, 10]
    vars['snr'] = snrs

    cell_counts = [2, 5, 10, 15, 25, 50, 75, 100, 125, 150, 200, 250, 300]
    vars['cell_count'] = cell_counts

    template =  {
        'negbin_prob': 0.5,
        'dispersion': 0.1,
        'cell_count': 180,
        'snr': 0.5,
        }

    data_cases = {}
    for vn, levels in vars.items():
        from copy import deepcopy
        for level in levels:
            dc = deepcopy(template)
            dc[vn] = level
            data_cases[f'{vn}: {level}'] = dc


    repeats = 5
    parameter_sets = []
    from copy import deepcopy
    for data_case, parameters in data_cases.items():
        parameters['data_case'] = data_case
        for ii in range(repeats):
            parameters['replicate'] = ii
            parameter_sets.append(deepcopy(parameters))


    # parameter_sets = parameter_sets[ : 5]  # Debug


    # preprocessing_options = {
    #         'cell normalised': [False, True],
    #         # 'read normalised': [False, True],
    #         'read normalised': [False],
    #         'transform 1': ['none', 'log1p'],
    #         'transform 2': ['none', 'zscores'],
    #         # 'pseudo_bulk': [False, 1, 2, 3, 5, 10],
    #         'pseudo_bulk': [False, 5],
    #         'shuffle': [False],
    #         'compute differences': [False, True],
    #         }
    preprocessing_options = {
            'cell normalised': [True],
            'read normalised': [False],
            'transform 1': ['log1p'],
            'transform 2': ['zscores'],
            'pseudo_bulk': [False],
            'shuffle': [False],
            'compute differences': [False],
            }





    from grn_code import functions

    def lasso_factory(density):
        def f(*args, **kwargs):
            return functions.lasso_T_density(
                    *args, **kwargs, target_density=density
                    )
        return f
    densities = [1, 3, 5, 10, 20, 50]
    lassos = [lasso_factory(density) for density in densities]

    inference_functions = [
            functions.fast_methods_inference,
            functions.random_inference,
            functions.correlation_inference,
            functions.perfect_inference,
            # *lassos,
            # functions.zscore_max_variants,
            # functions.zscore_without_controls,
            # functions.lsco_T_without_controls,
            # functions.inspre_inference,
            # functions.inspre_inference_hdf5,
            # functions.psgrn_inference,
            # functions.deepsem_inference,
            # functions.dspin_inference,
            # functions.dspin_inference_wrapper,
            # functions.genie3_inference,
            ]



    config = {
            'preprocessing_options': preprocessing_options,
            'inference_functions': inference_functions,
            'parameter_sets': parameter_sets,
            }
    fts = [repr(f) for f in inference_functions]
    config_for_serialisation = {
            'preprocessing_options': preprocessing_options,
            'parameter_sets': parameter_sets,
            'inference_functions__traces_for_serialisation': fts,
            }

    return config, config_for_serialisation




