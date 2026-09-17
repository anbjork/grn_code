
import anton_util


parameter_values = {
    'negbin_prob': [0.5],
    'cell_count': [50, 200],
    'dispersion': [0.1, 20],
    'snr': [0.03, 0.5],
    }
def recursive_combos(parameter_values, determined):
    from copy import deepcopy
    if len(parameter_values) == 0:
        return [determined]
    full_sets = []
    k, options = parameter_values.popitem()
    for option in options:
        determined[k] = option
        full_sets.extend(recursive_combos(
                deepcopy(parameter_values), deepcopy(determined)
                ))
    return full_sets
parameter_factorial_design = recursive_combos(parameter_values, {})
from pprint import pprint
pprint(parameter_factorial_design)


repeats = 5
parameter_sets = []
from copy import deepcopy
for parameter_set in parameter_factorial_design:
    for ii in range(repeats):
        parameter_set['replicate'] = ii
        parameter_sets.append(deepcopy(parameter_set))




from grn_code.data_simulation.configuration_imports import initialise_simulations
simulation_specifications = initialise_simulations(parameter_sets = parameter_sets)

simulation_specifications = simulation_specifications[ : 5] # Debug

from grn_code.pipeline_code import pipeline_base_path
anton_util.pickle_object(
        simulation_specifications,
        f'{pipeline_base_path}/simulation_specifications.pkl')


