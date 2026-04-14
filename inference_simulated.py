from pathlib import Path
import anton_util
import functions
import copy

path = Path('data/simulated/preprocessed.pkl')
data_sources = anton_util.unpickle_object(path)

estimated_networks = []
for ii, data_source in enumerate(data_sources):
    anton_util.log_timestamp(f'dataset {ii}...')
    ens = functions.run_inference_on_data(data=data_source)
    for method, en in ens.items():
        meta = copy.deepcopy(data_source['meta'])
        meta['method'] = method
        estimated_networks.append({
            'meta': meta,
            'estimated_network': en,
        })

output_dir = Path('inferences/simulated')
output_dir.mkdir(exist_ok=True, parents=True)
anton_util.pickle_object(
    estimated_networks,
    output_dir / f'estimated_networks.pkl')
anton_util.log_timestamp('Estimated networks saved.')



