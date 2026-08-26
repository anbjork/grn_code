
from grn_code.pipeline_functions import inference
from grn_code import functions

from grn_code.pipeline_configuration import pipeline_base_path as base_path
(base_path / 'inferences.pkl').unlink(missing_ok=True)

functions = [
    # functions.fast_methods_inference,
    # functions.random_inference,
    # functions.correlation_inference,
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

for method_function in functions:
    inference(
        data_path = base_path / 'data_processed.pkl',
        output_path = base_path / 'inferences.pkl',
        method_function = method_function
        )

