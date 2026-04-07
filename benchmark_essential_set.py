from pathlib import Path
import anton_util


benchmark_output_dir = Path('benchmarks/replogle')
[d.mkdir(exist_ok=True, parents=True) for d in [benchmark_output_dir]]


inference_dir = Path('inferences/replogle')
anton_util.log_timestamp('Loading estimated networks...')
estimated_networks = anton_util.unpickle_object(
    inference_dir / 'estimated_networks.pkl')
# anton_util.log_timestamp('Estimated networks loaded.')

reference_networks = {}
anton_util.log_timestamp('Loading reference networks...')
reference_networks = anton_util.unpickle_object(
    'data/replogle/compiled_reference_networks.pkl'
    )
# anton_util.log_timestamp('Reference networks loaded.')

from functions import benchmark_method_against_reference
stats = []
anton_util.log_timestamp('benchmarking')
for ii, data_source in enumerate(estimated_networks):
    anton_util.log_timestamp(f'dataset {ii}...')
    # Not the clearest naming. Structure of estimated networks is a list
    # of data sources, with dicts of methods
    mstats = {}
    for method, estimated_network in data_source.items():
        anton_util.log_timestamp(f'method {method}...')
        mstats[method] = {}
        for ref_name, ref_net in reference_networks.items():
            anton_util.log_timestamp(f'reference {ref_name}...')

            mstats[method][ref_name] = benchmark_method_against_reference(
                method = method,
                estimated_network = estimated_network,
                ref_name = ref_name,
                reference_network = ref_net,
                benchmark_output_dir = benchmark_output_dir,
                )
    stats.append(mstats)

anton_util.log_timestamp('Saving stats...')
anton_util.pickle_object(stats, benchmark_output_dir / 'stats.pkl')
anton_util.log_timestamp('Stats saved.')




