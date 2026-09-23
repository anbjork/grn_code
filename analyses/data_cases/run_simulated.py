
import anton_util

from grn_code.paths_anchor import output_base_path
output_path = output_base_path / 'data_cases'

# make_config = False
make_config = True

# read_simulation_specifications = False
read_simulation_specifications = True

config_name = 'configuration.pkl'
p = output_path / config_name
if make_config:
    from configuration import configure
    config = configure()
    output_path.mkdir(exist_ok=True, parents=True)
    anton_util.pickle_object(config, p)
else:
    try:
        config = anton_util.unpickle_object(p)
    except FileNotFoundError:
        print(f'{p} not found')
        import sys
        sys.exit(1)

from grn_code.pipeline_code import run_pipeline
run_pipeline(
        config = config,
        output_base_path = output_path,
        read_simulation_specifications = read_simulation_specifications)



from plot_by_metric import plot_metrics
plot_metrics(output_path = output_path)

from plot_by_data_case import plot_metrics
plot_metrics(output_path = output_path)


