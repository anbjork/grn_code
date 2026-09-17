
from grn_code.pipeline_functions import inference
from grn_code.pipeline_configuration import inference_functions

from grn_code.pipeline_configuration import pipeline_base_path as base_path
(base_path / 'inferences.pkl').unlink(missing_ok=True)


for method_function in inference_functions:
    inference(
        data_path = base_path / 'data_processed.pkl',
        output_path = base_path / 'inferences.pkl',
        method_function = method_function
        )

