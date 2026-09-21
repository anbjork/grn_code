from grn_code.pipeline_code import append_pickle
import anton_util


def append_results_with_pipeline(
        pipeline_base_path,
        output_base_path,
        ):

    anton_util.log_timestamp('appending to results...')
    outputs = [
            'simulation_specifications',
            'simulations',
            'preprocessed_data',
            'reference_networks',
            'inferences',
            'benchmarks',
            ]

    for output in outputs:
        anton_util.log_timestamp(f'{output}...')
        output_in_pipeline = anton_util.unpickle_object(
            f'{pipeline_base_path}/{output}.pkl'
            )
        append_pickle(
            output_in_pipeline,
            f'{output_base_path}/simulated/{output}.pkl',
            )




