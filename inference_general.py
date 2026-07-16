
from pathlib import Path
import anton_util
import copy

def inference(
        data_path,
        output_dir_path,
        method_name,
        method_function):

    path = data_path
    data_sources = anton_util.unpickle_object(path)

    estimated_networks = []
    for ii, data_source in enumerate(data_sources):
        anton_util.log_timestamp(f'dataset {ii}...')
        en = method_function(data=data_source)

        anton_util.log_timestamp(f'method {method_name}...')
        meta = copy.deepcopy(data_source['meta'])
        meta['method'] = method_name
        estimated_networks.append({
            'meta': meta,
            'estimated_network': en,
        })

    output_dir = Path(output_dir_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    p = output_dir / f'estimated_networks.pkl'
    if p.exists():
        oldnets = anton_util.unpickle_object(p)
        estimated_networks = oldnets + estimated_networks
    anton_util.pickle_object(
        estimated_networks,
        output_dir / f'estimated_networks.pkl')
    anton_util.log_timestamp('Estimated networks saved.')










