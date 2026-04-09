from pathlib import Path
import anton_util
import functions

path = Path('data/simulated/preprocessed.pkl')
data = anton_util.unpickle_object(path)

estimated_networks = []
for ii, elem in enumerate(data):
    anton_util.log_timestamp(f'dataset {ii}...')
    estimated_networks.append(functions.run_inference_on_data(data=elem))

output_dir = Path('inferences/simulated')
output_dir.mkdir(exist_ok=True, parents=True)
anton_util.pickle_object(
    estimated_networks,
    output_dir / f'estimated_networks.pkl')
anton_util.log_timestamp('Estimated networks saved.')



