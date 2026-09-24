


def configure():


    data_cases = {
            'easy': {
                'negbin_prob': 0.5,
                'dispersion': 0.1,
                'cell_count': 125,
                'snr': 0.5,
                },
    #         'low snr 0.05, old': {
    #             'negbin_prob': 0.5,
    #             'dispersion': 0.1,
    #             'cell_count': 125,
    #             'snr': 0.05,
    #             },
    #         'low snr 0.1': {
    #             'negbin_prob': 0.5,
    #             'dispersion': 0.1,
    #             'cell_count': 125,
    #             'snr': 0.1,
    #             },
    #         'low snr 0.3': {
    #             'negbin_prob': 0.5,
    #             'dispersion': 0.1,
    #             'cell_count': 125,
    #             'snr': 0.3,
    #             },
            'low snr 0.03': {
                'negbin_prob': 0.5,
                'dispersion': 0.1,
                'cell_count': 125,
                'snr': 0.03,
                },
    #         'low snr 0.035': {
    #             'negbin_prob': 0.5,
    #             'dispersion': 0.1,
    #             'cell_count': 125,
    #             'snr': 0.035,
    #             },
    #         'low snr 0.04': {
    #             'negbin_prob': 0.5,
    #             'dispersion': 0.1,
    #             'cell_count': 125,
    #             'snr': 0.04,
    #             },
    #         'low snr 0.045': {
    #             'negbin_prob': 0.5,
    #             'dispersion': 0.1,
    #             'cell_count': 125,
    #             'snr': 0.045,
    #             },
    #         'high dropout 10, old': {
    #             'negbin_prob': 0.5,
    #             'dispersion': 10,
    #             'cell_count': 125,
    #             'snr': 0.5,
    #             },
    #         'high dropout 50': {
    #             'negbin_prob': 0.5,
    #             'dispersion': 50,
    #             'cell_count': 125,
    #             'snr': 0.5,
    #             },
            'high dropout 20': {
                'negbin_prob': 0.5,
                'dispersion': 20,
                'cell_count': 125,
                'snr': 0.5,
                },
    #         'high dropout 15': {
    #             'negbin_prob': 0.5,
    #             'dispersion': 15,
    #             'cell_count': 125,
    #             'snr': 0.5,
    #             },
            }

    # # This adjusted for the genesnake version. snrs are a bit different scale
    # # for this one
    # data_cases = {
    #         'easy': {
    #             'negbin_prob': 0.5,
    #             'dispersion': 0.1,
    #             'cell_count': 125,
    #             'snr': 10,
    #             },
    #         'low snr': {
    #             'negbin_prob': 0.5,
    #             'dispersion': 0.1,
    #             'cell_count': 125,
    #             'snr': 1,
    #             },
    #         'high dropout': {
    #             'negbin_prob': 0.5,
    #             'dispersion': 10,
    #             'cell_count': 125,
    #             'snr': 10,
    #             },
    #         }




    # template =  {
    #     'negbin_prob': 0.5,
    #     'dispersion': 0.1,
    #     'cell_count': 125,
    #     'snr': 0.5,
    #     }
    # old_dispersions = [0.1, 10, 15, 20, 50]
    # additional_dispersions = [0.2, 0.3, 0.5, 1, 2, 3, 5, 7]
    # even_more = [30, 40]
    # all_dispersions = old_dispersions + additional_dispersions + even_more
    # data_cases = {}
    # from copy import deepcopy
    # for dispersion in all_dispersions:
    #     dc = deepcopy(template)
    #     dc['dispersion'] = dispersion
    #     data_cases[f'high dropouts {dispersion}'] = dc


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
    inference_functions = [
            functions.fast_methods_inference,
            functions.random_inference,
            functions.correlation_inference,
            functions.perfect_inference,
            # functions.zscore_max_variants,
            # functions.zscore_without_controls,
            # functions.lsco_T_without_controls,
            # functions.inspre_inference,
            # functions.inspre_inference_hdf5,
            # functions.psgrn_inference,
            # functions.genie3_inference,
            # functions.deepsem_inference,
            # functions.dspin_inference,
            # functions.dspin_inference_wrapper,
            ]



    config = {
            'preprocessing_options': preprocessing_options,
            'inference_functions': inference_functions,
            'parameter_sets': parameter_sets,
            }


    return config

