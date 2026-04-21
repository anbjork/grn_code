
from pathlib import Path
import anton_util
import functions
import copy

path = Path('data/simulated/preprocessed.pkl')
data_sources = anton_util.unpickle_object(path)

# # Debug
# data_sources = data_sources[:1]

estimated_networks = []
for ii, data_source in enumerate(data_sources):
    anton_util.log_timestamp(f'dataset {ii}...')
    en = functions.genie3_inference(data=data_source)

    meta = copy.deepcopy(data_source['meta'])
    meta['method'] = 'genie3'
    estimated_networks.append({
        'meta': meta,
        'estimated_network': en,
    })

output_dir = Path('inferences/simulated')
output_dir.mkdir(exist_ok=True, parents=True)
p = output_dir / f'estimated_networks.pkl'
if p.exists():
    oldnets = anton_util.unpickle_object(p)
    estimated_networks = oldnets + estimated_networks
anton_util.pickle_object(
    estimated_networks,
    output_dir / f'estimated_networks.pkl')
anton_util.log_timestamp('Estimated networks saved.')


