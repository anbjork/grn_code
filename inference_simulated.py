
from pathlib import Path
from pipeline_functions import inference
import functions

base_path = Path('outputs__in_pipeline/simulated')
(base_path / 'inferences.pkl').unlink(missing_ok=True)

functions = [
    # functions.fast_methods_inference,
    # functions.random_inference,
    # functions.correlation_inference,
    # functions.inspre_inference,
    # functions.inspre_inference_hdf5,
    functions.psgrn_inference,
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

