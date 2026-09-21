
import anton_util

from grn_code.paths_anchor import output_base_path
output_path = output_base_path / 'some_other_analysis'

make_config = False
# make_config = True

# read_simulation_specifications = False
read_simulation_specifications = True


config_name = 'configuration.pkl'
if make_config:
    from configuration import configure
    config = configure(output_path = output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    anton_util.pickle_object(config, output_path / config_name)
else:
    config = anton_util.unpickle_object(output_path / config_name)


from grn_code.pipeline_code import run_pipeline
run_pipeline(
        config = config, 
        output_base_path = output_path, 
        read_simulation_specifications = read_simulation_specifications)



from plot_metrics import plot_metrics
plot_metrics(output_path = output_path)




