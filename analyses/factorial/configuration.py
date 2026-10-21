

from collections.abc import Callable
from typing import Any
InferenceMethod = Callable[[Any], dict[str, Any]]



def configure():


    parameter_values = {
        'negbin_prob': [0.5],
        'cell_count': [10, 250],
        'dispersion': [0.1, 40],
        'snr': [0.02, 1],
        }
    def recursive_combos(parameter_values, determined):
        from copy import deepcopy
        if len(parameter_values) == 0:
            return [determined]
        full_sets = []
        k, options = parameter_values.popitem()
        for option in options:
            determined[k] = option
            full_sets.extend(recursive_combos(
                    deepcopy(parameter_values), deepcopy(determined)
                    ))
        return full_sets
    parameter_factorial_design = recursive_combos(parameter_values, {})
    from pprint import pprint
    pprint(parameter_factorial_design)


    repeats = 5
    parameter_sets = []
    from copy import deepcopy
    for parameter_set in parameter_factorial_design:
        for ii in range(repeats):
            parameter_set['replicate'] = ii
            parameter_sets.append(deepcopy(parameter_set))





    preprocessing_options = {
            'cell normalised': [False, True],
            # 'read normalised': [False, True],
            'read normalised': [False],
            'transform 1': ['none', 'log1p'],
            'transform 2': ['none', 'zscores'],
            # 'pseudo_bulk': [False, 1, 2, 3, 5, 10],
            'pseudo_bulk': [False, 5],
            'shuffle': [False],
            'compute differences': [False, True],
            }
    # preprocessing_options = {
    #         'cell normalised': [True],
    #         'read normalised': [False],
    #         'transform 1': ['log1p'],
    #         'transform 2': ['zscores'],
    #         'pseudo_bulk': [False],
    #         'shuffle': [False],
    #         'compute differences': [False],
    #         }




    from grn_code import inference_functions as functions

    def lasso_factory(density):
        def f(*args, **kwargs):
            return functions.lasso_T_density(
                    *args, **kwargs, target_density=density
                    )
        return f
    # densities  = [0.3, 1, 10]
    densities  = [1]
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
            # functions.zscore_max_variants,
            # functions.zscore_without_controls,
            # functions.lsco_T_without_controls,
            # functions.inspre_inference,
            # functions.inspre_inference_hdf5,
            functions.psgrn_inference,
            *lassos,
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


