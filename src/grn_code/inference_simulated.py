
from grn_code.pipeline_code import inference
from grn_code.pipeline_code import pipeline_base_path
(pipeline_base_path / 'inferences.pkl').unlink(missing_ok=True)

# from grn_code.pipeline_configuration import inference_functions
import anton_util
config = anton_util.unpickle_object(pipeline_base_path / 'pipeline_configuration.pkl')
inference_functions = config['inference_functions']

from grn_code import functions
fs = []
for fn in inference_functions:
    fs.append(getattr(functions, fn))

for inference_function in fs:
    inference(
        data_path = pipeline_base_path / 'preprocessed_data.pkl',
        output_path = pipeline_base_path / 'inferences.pkl',
        method_function = inference_function,
        )

