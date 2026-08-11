import anton_util
from grn_code.functions import benchmark_method_against_reference

from grn_code.pipeline_configuration import pipeline_base_path as base_path

inferred = anton_util.unpickle_object(
    base_path / 'inferences.pkl'
    )
reference_networks = anton_util.unpickle_object(
    base_path / 'reference_networks.pkl'
    )
reference_networks_dict = {
        # frozensets are immutable, and so can be used as keys.
        # This way, it matches keys on all meta data fields, without
        # me needing to construct those keys manually.
        # Guarantees only matching if all meta data fits,
        # so no risk of errors
        # (Well, as long as meta data is complete, to uniquely identify)
        frozenset(r['meta']['dataset_parameters'].items()): r['data'] 
        for r in reference_networks
        }

anton_util.log_timestamp('benchmarking...')
stats = []
for ii, inference in enumerate(inferred):
    meta = inference['meta']
    estimated_network = inference['estimated_network']
    anton_util.log_timestamp(f'inference {ii}...')
    if estimated_network is None:
        print('estimated networks is None')
        print(f'inference error recorded: {meta["inference error"]}')
        mstats = None
        stats.append({
            'meta': meta,
            'data': mstats,
        })
        continue
    meta_key = frozenset(meta['dataset_parameters'].items())
    reference_network = reference_networks_dict[meta_key]
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



