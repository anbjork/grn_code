

negbin_prob = 0.5
cell_count = 180



from collections.abc import Callable
from typing import Any
InferenceMethod = Callable[[Any], dict[str, Any]]




def configure():


    data_cases = {
            'easy': {
                'negbin_prob': negbin_prob,
                'dispersion': 0.1,
                'cell_count': cell_count,
                'snr': 0.5,
                },
            # 'less easy': {
            #     'negbin_prob': negbin_prob,
            #     'dispersion': 1.5,
            #     'cell_count': cell_count,
            #     'snr': 0.2,
            #     },
            # 'less easy; low snr 0.03': {
            #     'negbin_prob': negbin_prob,
            #     'dispersion': 1.5,
            #     'cell_count': cell_count,
            #     'snr': 0.03,
            #     },
            # 'less easy; high dropout 20': {
            #     'negbin_prob': negbin_prob,
            #     'dispersion': 20,
            #     'cell_count': cell_count,
            #     'snr': 0.2,
            #     },
            # 'low snr 0.03': {
            #     'negbin_prob': negbin_prob,
            #     'dispersion': 0.1,
            #     'cell_count': cell_count,
            #     'snr': 0.03,
            #     },
            # 'high dropout 20': {
            #     'negbin_prob': negbin_prob,
            #     'dispersion': 20,
            #     'cell_count': cell_count,
            #     'snr': 0.5,
            #     },
            # 'hard': {
            #     'negbin_prob': negbin_prob,
            #     'dispersion': 20,
            #     'cell_count': cell_count,
            #     'snr': 0.03,
            #     },
            'lower snr 0.015': {
                'negbin_prob': negbin_prob,
                'dispersion': 0.1,
                'cell_count': cell_count,
                'snr': 0.015,
                },
            'higher dropout 70': {
                'negbin_prob': negbin_prob,
                'dispersion': 70,
                'cell_count': cell_count,
                'snr': 0.5,
                },
            'hard': {
                'negbin_prob': negbin_prob,
                'dispersion': 70,
                'cell_count': cell_count,
                'snr': 0.015,
                },
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




    repeats = 10
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
            'transform 2': ['none', 'zscores'],
            # 'transform 2': ['zscores'],
            'pseudo_bulk': [False],
            'shuffle': [False],
            # 'compute differences': [False, True],
            'compute differences': [False],
            }




    from grn_code import inference_functions as functions

    def lasso_factory(density):
        def f(*args, **kwargs):
            return functions.lasso_T_density(
                    *args, **kwargs, target_density=density
                    )
        return f
    densities  = [0.3, 1, 10]
    lassos = [lasso_factory(density) for density in densities]

    inference_functions: list[InferenceMethod] = [
            functions.fast_methods_inference,
            functions.random_inference,
            functions.correlation_inference,
            # This one has a different signature, which is handled in the pipeline.
            # It cheats by taking the reference networks as input
            # That's why pyright gets confused about it's signature,
            # and is ignored
            functions.perfect_inference,  # pyright: ignore
            *lassos,
            # functions.zscore_max_variants,
            # functions.zscore_without_controls,
            # functions.lsco_T_without_controls,
            # functions.inspre_inference,
            # functions.inspre_inference_hdf5,
            functions.psgrn_inference,
            # functions.deepsem_inference,
            # functions.dspin_inference,
            functions.dspin_inference_wrapper,
            functions.bigsm_inference,
            functions.regdiffusion_inference,
            functions.genie3_inference,
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

