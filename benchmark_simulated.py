from pathlib import Path
import anton_util
from functions import benchmark_method_against_reference

base_path = Path('outputs__in_pipeline/simulated')

inferred = anton_util.unpickle_object(
    base_path / 'inferences.pkl'
    )
ground_truths = anton_util.unpickle_object(
    base_path / 'reference_networks.pkl'
    )

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

outfile = base_path / 'benchmarks.pkl'
outfile.parent.mkdir(parents = True, exist_ok = True)
anton_util.pickle_object(stats, outfile)



