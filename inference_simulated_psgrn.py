from pathlib import Path
from inference_general import inference
import functions

inference(
    data_path = Path('data/simulated/preprocessed.pkl'),
    output_dir_path = Path('inferences/simulated'),
    method_name = 'psgrn_all_bugfix',
    method_function = functions.psgrn_inference
    )



