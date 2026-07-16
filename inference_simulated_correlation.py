from pathlib import Path
from inference_general import inference
import functions

inference(
        data_path = Path('data/simulated/preprocessed.pkl'),
        output_dir_path = Path('inferences/simulated'),
        method_name = 'correlation',
        method_function = functions.correlation_inference
        )



