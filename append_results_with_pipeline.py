from pipeline_functions import append_pickle
import anton_util

anton_util.log_timestamp('appending to results...')
outputs = [
        'data_processed',
        'reference_networks',
        'inferences',
        'benchmarks',
        ]

for output in outputs:
    anton_util.log_timestamp(f'{output}...')
    output_in_pipeline = anton_util.unpickle_object(
        f'outputs__in_pipeline/simulated/{output}.pkl'
        )
    append_pickle(
        output_in_pipeline,
        f'outputs/simulated/{output}.pkl',
        )




