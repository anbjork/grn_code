

negbin_prob = 0.5
cell_count = 180


def configure():


    data_cases = {
            'easy': {
                'negbin_prob': negbin_prob,
                'dispersion': 0.1,
                'cell_count': cell_count,
                'snr': 0.5,
                },
    #         'low snr 0.05, old': {
    #             'negbin_prob': negbin_prob,
    #             'dispersion': 0.1,
    #             'cell_count': cell_count,
    #             'snr': 0.05,
    #             },
    #         'low snr 0.1': {
    #             'negbin_prob': negbin_prob,
    #             'dispersion': 0.1,
    #             'cell_count': cell_count,
    #             'snr': 0.1,
    #             },
    #         'low snr 0.3': {
    #             'negbin_prob': negbin_prob,
    #             'dispersion': 0.1,
    #             'cell_count': cell_count,
    #             'snr': 0.3,
    #             },
            'low snr 0.03': {
                'negbin_prob': negbin_prob,
                'dispersion': 0.1,
                'cell_count': cell_count,
                'snr': 0.03,
                },
    #         'low snr 0.035': {
    #             'negbin_prob': negbin_prob,
    #             'dispersion': 0.1,
    #             'cell_count': cell_count,
    #             'snr': 0.035,
    #             },
    #         'low snr 0.04': {
    #             'negbin_prob': negbin_prob,
    #             'dispersion': 0.1,
    #             'cell_count': cell_count,
    #             'snr': 0.04,
    #             },
    #         'low snr 0.045': {
    #             'negbin_prob': negbin_prob,
    #             'dispersion': 0.1,
    #             'cell_count': cell_count,
    #             'snr': 0.045,
    #             },
    #         'high dropout 10, old': {
    #             'negbin_prob': negbin_prob,
    #             'dispersion': 10,
    #             'cell_count': cell_count,
    #             'snr': 0.5,
    #             },
    #         'high dropout 50': {
    #             'negbin_prob': negbin_prob,
    #             'dispersion': 50,
    #             'cell_count': cell_count,
    #             'snr': 0.5,
    #             },
            'high dropout 20': {
                'negbin_prob': negbin_prob,
                'dispersion': 20,
                'cell_count': cell_count,
                'snr': 0.5,
                },
    #         'high dropout 15': {
    #             'negbin_prob': negbin_prob,
    #             'dispersion': 15,
    #             'cell_count': cell_count,
    #             'snr': 0.5,
    #             },
            }

    # # This adjusted for the genesnake version. snrs are a bit different scale
    # # for this one
    # data_cases = {
    #         'easy': {
    #             'negbin_prob': negbin_prob,
    #             'dispersion': 0.1,
    #             'cell_count': cell_count,
    #             'snr': 10,
    #             },
    #         'low snr': {
    #             'negbin_prob': negbin_prob,
    #             'dispersion': 0.1,
    #             'cell_count': cell_count,
    #             'snr': 1,
    #             },
    #         'high dropout': {
    #             'negbin_prob': negbin_prob,
    #             'dispersion': 10,
    #             'cell_count': cell_count,
    #             'snr': 10,
    #             },
    #         }




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
            # 'transform 2': ['none', 'zscores'],
            'transform 2': ['zscores'],
            'pseudo_bulk': [False],
            'shuffle': [False],
            'compute differences': [False],
            }




    from grn_code import functions

#    def lasso_factory(density):
       #def f(*args, **kwargs):
           #return functions.lasso_T_density(
                   #*args, **kwargs, target_density=density
                   #)
       #return f

    # densities = [1, 3, 5, 10, 20, 50]
   #densities = [0.01, 0.05, 0.1, 0.3, 0.5, 1, 1.5, 2, 3]
    #assos = [lasso_factory(density) for density in densities]

    inference_functions = [
             functions.fast_methods_inference,
             functions.random_inference,
             functions.correlation_inference,
             functions.perfect_inference,
            #lassos,
             functions.zscore_max_variants,
             functions.zscore_without_controls,
             functions.lsco_T_without_controls,
            # functions.inspre_inference,
            # functions.inspre_inference_hdf5,
            # functions.psgrn_inference,
            # functions.deepsem_inference,
            # functions.dspin_inference,
            # functions.dspin_inference_wrapper,
            # functions.genie3_inference,
            functions.bigsm_inference,
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

