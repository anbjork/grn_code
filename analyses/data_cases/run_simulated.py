
import anton_util

from grn_code.paths_anchor import output_base_path
output_path = output_base_path / 'data_cases'

# read_simulation_specifications = False
read_simulation_specifications = True

config_name = 'configuration.pkl'
p = output_path / config_name
from configuration import configure
config, config_for_serialization = configure()
output_path.mkdir(exist_ok=True, parents=True)
anton_util.pickle_object(config_for_serialization, p)

from grn_code.pipeline_code import run_pipeline
run_pipeline(
        config = config,
        output_base_path = output_path,
        read_simulation_specifications = read_simulation_specifications)

from plot_results import plot_metrics
plot_metrics(output_path, major_col='metric', minor_col='data_case')
plot_metrics(output_path, major_col='data_case', minor_col='metric')


