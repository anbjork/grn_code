from pathlib import Path
import anton_util
import functions
import copy

indir = Path('data/replogle')
data_set_name = 'K562_essential_raw_singlecell_01'

anton_util.log_timestamp(f'Loading {data_set_name}...')
data_sources = anton_util.unpickle_object(
    f'{indir}/{data_set_name}_preprocessed_2.pkl')

# # Debug
# print(data_sources)

estimated_networks = []
for ii, data_source in enumerate(data_sources):
    anton_util.log_timestamp(f'dataset {ii}...')
    en = functions.dspin_inference(data=data_source)

    meta = copy.deepcopy(data_source['meta'])
    meta['method'] = 'dspin'
    estimated_networks.append({
        'meta': meta,
        'estimated_network': en,
    })

output_dir = Path('inferences/replogle')
output_dir.mkdir(exist_ok=True, parents=True)

p = output_dir / f'estimated_networks.pkl'
if p.exists():
    oldnets = anton_util.unpickle_object(p)
    estimated_networks = oldnets + estimated_networks
anton_util.log_timestamp('saving estimated networks...')
anton_util.pickle_object(
    estimated_networks,
    output_dir / f'estimated_networks.pkl')





