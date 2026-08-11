from grn_code.pipeline_functions import append_pickle
import anton_util

from grn_code.pipeline_configuration import pipeline_base_path
from grn_code.pipeline_configuration import output_base_path

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
        f'{pipeline_base_path}/simulated/{output}.pkl'
        )
    append_pickle(
        output_in_pipeline,
        f'{output_base_path}/simulated/{output}.pkl',
        )




