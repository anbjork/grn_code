from pathlib import Path
import anton_util

benchmark_output_dir = Path('benchmarks/simulated')
[d.mkdir(exist_ok=True, parents=True) for d in [benchmark_output_dir]]

inference_dir = Path('inferences/simulated')
inferred = anton_util.unpickle_object(
    inference_dir / 'estimated_networks.pkl')

# path = Path('data/simulated/preprocessed.pkl')
# data_sources = anton_util.unpickle_object(path)


ground_truths = anton_util.unpickle_object('data/simulated/ground_truths.pkl')

from functions import benchmark_method_against_reference
anton_util.log_timestamp('benchmarking...')
stats = []
for ii, inference in enumerate(inferred):
    meta = inference['meta']
    estimated_network = inference['estimated_network']
    anton_util.log_timestamp(f'inference {ii}...')
    reference_network = ground_truths[meta['replicate']]
    mstats = benchmark_method_against_reference(
        method = meta['method'],
        estimated_network = estimated_network,
        reference_network = reference_network,
        benchmark_output_dir = benchmark_output_dir,
        )
    stats.append({
        'meta': meta,
        'data': mstats,
    })

anton_util.pickle_object(stats, benchmark_output_dir / 'stats.pkl')


