from pathlib import Path
import anton_util

benchmark_output_dir = Path('benchmarks/simulated')
[d.mkdir(exist_ok=True, parents=True) for d in [benchmark_output_dir]]

inference_dir = Path('inferences/simulated')
inferred = anton_util.unpickle_object(
    inference_dir / 'estimated_networks.pkl')

path = Path('data/simulated/preprocessed.pkl')
data_sources = anton_util.unpickle_object(path)

from functions import benchmark_method_against_reference
stats = []
for data_source, estimated_networks in zip(data_sources, inferred):
    reference_network = data_source['A']
    mstats = {}
    for method, estimated_network in estimated_networks.items():
        data_id = f'{data_source["index"]}__shuffled_{data_source["shuffle"]}'
        anton_util.log_timestamp(f'Benchmarking {method} for {data_id}...')
        mstats[method] = benchmark_method_against_reference(
            method = method,
            estimated_network = estimated_network,
            ref_name = 'A',
            reference_network = reference_network,
            benchmark_output_dir = benchmark_output_dir,
            )
        # anton_util.log_timestamp(f'Finished benchmarking {method} for {data_id}')
    stats.append(mstats)

anton_util.pickle_object(stats, benchmark_output_dir / 'stats.pkl')


