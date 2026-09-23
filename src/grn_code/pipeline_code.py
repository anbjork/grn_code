
import anton_util
import json
import subprocess
from datetime import datetime


def save_run_metadata(output_path):
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True
    ).strip()

    metadata = {
        'git_commit': commit,
        'timestamp': datetime.now().isoformat(),
    }

    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / 'run_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


def run_pipeline(
        config,
        output_base_path,
        read_simulation_specifications,
        ):

    def handle_perfect_inference(config):
        def bind_location_of_cheat(config):
            previous_inference_function = config['inference_functions'][ii]
            def perfect_inference(data):
                return previous_inference_function(
                        data,
                        location_of_cheat_sheet = pipeline_output_path,
                        )
            config['inference_functions'][ii] = perfect_inference

        matches = [
                ii for ii, f in enumerate(config['inference_functions'])
                if f.__name__ == 'perfect_inference'
                ]
        if len(matches) == 0:
            pass
        elif len(matches) == 1:
            ii = matches[0]
            bind_location_of_cheat(config)
        else:
            raise ValueError('Multiple perfect_inference functions found')

    save_run_metadata(output_base_path)

    pipeline_output_path = output_base_path / 'in_pipeline'
    pipeline_output_path.mkdir(exist_ok=True, parents=True)

    if read_simulation_specifications:
        p = pipeline_output_path / 'simulation_specifications.pkl'
        try:
            simulation_specifications = anton_util.unpickle_object(p)
        except FileNotFoundError:
            print(f'{p} not found')
            import sys
            sys.exit(1)
    else:
        from grn_code.data_simulation import configuration_imports as ci
        simulation_specifications = ci.initialise_simulations(
                parameter_sets = config['parameter_sets'],
                base_path = pipeline_output_path,
                )
        anton_util.pickle_object(
                simulation_specifications,
                pipeline_output_path / 'simulation_specifications.pkl'
                )

    import grn_code.data_simulation.simulate_genespider as simulate_genespider
    simulate_genespider.main(
            output_path = pipeline_output_path,
            job_specifications = simulation_specifications,
            )

    import grn_code.data_simulation.gather_simulation_data as gather_simulation_data
    gather_simulation_data.main(
            output_path = pipeline_output_path,
            simulation_specifications = simulation_specifications,
            )

    import grn_code.preprocess_simulated_data_and_networks as psdn
    psdn.main(
            output_path = pipeline_output_path,
            preprocessing_options = config['preprocessing_options'],
            )

    handle_perfect_inference(config)
    from grn_code.inference_simulated import inference_simulated
    inference_simulated(
            output_path = pipeline_output_path,
            inference_functions = config['inference_functions'],
            )

    from grn_code.benchmark_simulated import benchmark_simulated
    benchmark_simulated(
            output_path = pipeline_output_path,
            )

    from grn_code.append_results_with_pipeline import append_results_with_pipeline
    append_results_with_pipeline(
            pipeline_base_path = pipeline_output_path,
            output_base_path = output_base_path,
            )

    from grn_code.compile_results_simulated import compile_results_simulated
    compile_results_simulated(
            output_path = output_base_path,
            )












def append_pickle(data, path):
    from pathlib import Path
    import anton_util
    p = Path(path)
    if p.exists():
        previous_data = anton_util.unpickle_object(path)
        data = previous_data + data
    else:
        p.parent.mkdir(exist_ok=True, parents=True)
    anton_util.pickle_object(data, path)
    anton_util.log_timestamp(f'total data length: {len(data)}')







def inference(
        data_path,
        output_path,
        method_function):

    from pathlib import Path
    import anton_util
    import copy

    f_name = method_function.__name__

    path = data_path
    data_sources = anton_util.unpickle_object(path)
    # data_sources = [data_sources[3]] # Debug

    estimated_networks = []
    for ii, data_source in enumerate(data_sources):
        anton_util.log_timestamp(f'dataset {ii}...')
        anton_util.log_timestamp(f_name)

        try:
            ens = method_function(data=data_source)
            error = None
        except Exception as e:
            error = repr(e)
            anton_util.log_timestamp(
                    f'Error in method {f_name}: {error}')
            ens = {f_name: None}
        # Debug
        # ens = method_function(data=data_source)
        # error = None

        for method_name, estimated_network in ens.items():
            meta = copy.deepcopy(data_source['meta'])
            meta['method'] = method_name
            meta['inference error'] = error
            estimated_networks.append({
                'meta': meta,
                'estimated_network': estimated_network,
                })

    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    anton_util.log_timestamp('Saving...')
    append_pickle(estimated_networks, output_path)










