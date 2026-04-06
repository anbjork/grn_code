from pathlib import Path
import anton_util
import functions


indir = Path('data/replogle')
indir.mkdir(exist_ok=True, parents=True)
data_set_name = 'K562_essential_raw_singlecell_01'

anton_util.log_timestamp(f'Loading {data_set_name}...')
data_sources = anton_util.unpickle_object(
    f'{indir}/{data_set_name}_preprocessed_2.pkl')
anton_util.log_timestamp(f'{data_set_name} loaded.')


estimated_networks = []
for data_source in data_sources:
    anton_util.log_timestamp(f'Processing data source: {data_source["shuffle"]}')
    estimated_networks.append(functions.run_inference_on_data(data=data_source))


output_dir = Path('inferences/replogle')
output_dir.mkdir(exist_ok=True, parents=True)

anton_util.log_timestamp('Saving estimated networks...')
anton_util.pickle_object(
    estimated_networks,
    output_dir / f'estimated_networks.pkl')
anton_util.log_timestamp('Estimated networks saved.')




