from pathlib import Path
import anton_util
import pipeline_functions

benchmark_output_dir = Path('benchmarks/simulated')
[d.mkdir(exist_ok=True, parents=True) for d in [benchmark_output_dir]]

inference_dir = Path('inferences/simulated')
inferred = anton_util.unpickle_object(
    inference_dir / 'estimated_networks.pkl')


ground_truths = anton_util.unpickle_object(
    'data_processed/simulated/ground_truths.pkl'
    )

from functions import benchmark_method_against_reference
anton_util.log_timestamp('benchmarking...')
stats = []
for ii, inference in enumerate(inferred):
    meta = inference['meta']
    estimated_network = inference['estimated_network']
    anton_util.log_timestamp(f'inference {ii}...')
    reference_network = ground_truths[meta['replicate']]
    try:
        mstats = benchmark_method_against_reference(
            method = meta['method'],
            estimated_network = estimated_network,
            reference_network = reference_network,
            )
    except Exception as e:
        anton_util.log_timestamp(f'Error benchmarking inference {ii} with method {meta["method"]}: {e}')
        anton_util.log_timestamp('Inference metadata:')
        anton_util.log_timestamp(f'{meta}')
        mstats = None
    stats.append({
        'meta': meta,
        'data': mstats,
    })

outfile = benchmark_output_dir / 'stats.pkl'
pipeline_functions.append_pickle(stats, outfile)


