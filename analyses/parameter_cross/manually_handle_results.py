
import anton_util

output_types = [
        'preprocessed_data',
        'reference_networks',
        'inferences',
        'benchmarks',
        'compiled_results',
        ]

from local_imports import output_path
# base_path = 'outputs/in_pipeline'
base_path = f'{output_path}/simulated'

outputs = []
for output_type in output_types:
    anton_util.log_timestamp(f'{output_type}...')
    try:
        outputs.append(anton_util.unpickle_object(
            f'{base_path}/{output_type}.pkl',
            ))
    except FileNotFoundError:
        print(f'File not found: {base_path}/{output_type}.pkl')
        outputs.append([])
anton_util.log_timestamp('done reading')



# Note; Only for convenience, and how contents of outputs are written to disk
# in the save function prepared below
preprocessed_data, reference_networks, inferences, benchmarks, compiled_results = outputs



def extract_meta_data(datasets):
    from copy import deepcopy
    metas = deepcopy([d['meta'] for d in datasets])
    for meta in metas:
        for k, v in meta['dataset_parameters'].items():
            meta[k] = v
        meta.pop('dataset_parameters')
    dataset_dict = {}
    for meta, data in zip(metas, datasets):
        dataset_dict[frozenset(meta.items())] = data
    return metas, dataset_dict







# The intention is to do whatever manual handling desired interactively here



# Checking dropouts in a convenient fashion
dropouts = []
for d in preprocessed_data:
    cols = {k: v for k, v in d['meta'].items() if 'fraction' in k}
    tmp = 'data_case'
    cols[tmp] = d['meta']['dataset_parameters'][tmp]
    dropouts.append(cols)
import pandas as pd
dropouts = pd.DataFrame(dropouts)









def save_results():
    anton_util.log_timestamp('saving results...')
    for output_type, output in zip(output_types, outputs):
        anton_util.pickle_object(
            output,
            f'outputs/simulated/{output_type}.pkl',
            )

