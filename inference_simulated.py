
from pathlib import Path
from pipeline_functions import inference
import functions

functions = [
    # functions.fast_methods_inference,
    functions.correlation_inference,
    # functions.psgrn_inference,
    # functions.genie3_inference,
    # functions.deepsem_inference,
    # functions.dspin_inference,
    ]

for method_function in functions:

    inference(
        data_path = Path('data_processed/simulated/preprocessed.pkl'),
        output_dir_path = Path('inferences/simulated'),
        method_function = method_function
        )



