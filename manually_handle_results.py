
import anton_util

anton_util.log_timestamp('appending to results...')
output_types = [
        'data_processed',
        'inferences',
        'benchmarks',
        ]

outputs = []
for output_type in output_types:
    anton_util.log_timestamp(f'{output_type}...')
    outputs.append(anton_util.unpickle_object(
        f'outputs/simulated/{output_type}.pkl',
        ))

data_processed, inferences, benchmarks = outputs



# The intention is to do whatever manual handling desired interactively here

def save_results():
    anton_util.log_timestamp('saving results...')
    for output_type, output in zip(output_types, outputs):
        anton_util.pickle_object(
            output,
            f'outputs/simulated/{output_type}.pkl',
            )

