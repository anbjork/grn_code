
from pathlib import Path
from pipeline_functions import inference
import functions

functions = [
    functions.fast_methods_inference,
    # functions.correlation_inference,
    # functions.psgrn_inference,
    # functions.genie3_inference,
    # functions.deepsem_inference,
    # functions.dspin_inference,
    ]

for method_function in functions:
    base_path = Path('outputs__in_pipeline/simulated')
    inference(
        data_path = base_path / 'data_processed.pkl',
        output_path = base_path / 'inferences.pkl',
        method_function = method_function
        )



