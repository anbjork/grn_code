
from pathlib import Path
import anton_util



def get_uuid():
    import uuid
    return str(uuid.uuid4())



def initialise_simulations(parameter_sets):

    sim_data = d / 'data'
    flag_files = d / 'simulation_completed_flags'
    for path in [sim_data, flag_files]:
        path.mkdir(exist_ok = True, parents = True)

    simulation_specifications = []
    for parameters in parameter_sets:

        tmp = [k + f'_{v}' for k, v in parameters.items()]
        parameter_tag = '__'.join(tmp)

        # NOTE: If you change this list, you must make the corresponding change
        # in simulate.m, otherwise mismatch bug.
        # It's because I could not be fucked to look up how to do a less
        # brittle approarch in Matlab.
        # It bit me once, which confirms my theory that poeople shouldn't
        # cut corners like this. As an experiment, let's see if it bites me again.
        simulation_matrix_names = [
            'A', 'Y', 'X', 'P', 'SCC', 'Ed', 'Eg'
            ]

        # PosixPath is not JSON serialisable, 
        # so converting paths to strings below
        files = {
            name: str(sim_data / get_uuid())
            for name in simulation_matrix_names
            }
        sim_flag_file = str(flag_files / get_uuid())
        job_specification = {
            'parameters': parameters,
            'parameter_tag': parameter_tag,
            'python_global_parameters_for_matlab': python_global_parameters,
            'simulation_matrix_names': simulation_matrix_names,
            'simulation_matrix_files': files,
            'simulation_completed_flag_file': sim_flag_file,
            }
        simulation_specifications.append(job_specification)


    # simulation_specifications = simulation_specifications[ : 5]


    return(simulation_specifications)



out_dir = Path('outputs')
d = Path('outputs/simulation')
for path in [out_dir, d]:
    path.mkdir(exist_ok = True, parents = True)

python_global_parameters = {
        'number_of_genes': 100,
        'average_network_degree': 3
        }

# data_cases = {
#         'easy': {
#             'negbin_prob': 0.5,
#             'dispersion': 0.1,
#             'cell_count': 125,
#             'snr': 0.5,
#             },
#         'low snr': {
#             'negbin_prob': 0.5,
#             'dispersion': 0.1,
#             'cell_count': 125,
#             'snr': 0.05,
#             },
#         'high dropout': {
#             'negbin_prob': 0.5,
#             'dispersion': 10,
#             'cell_count': 125,
#             'snr': 0.5,
#             },
#         }
data_cases = {
        'easy': {
            'negbin_prob': 0.5,
            'dispersion': 0.1,
            'cell_count': 125,
            'snr': 0.5,
            },
        # 'low snr': {
        #     'negbin_prob': 0.5,
        #     'dispersion': 0.1,
        #     'cell_count': 125,
        #     'snr': 0.05,
        #     },
        # 'high dropout': {
        #     'negbin_prob': 0.5,
        #     'dispersion': 10,
        #     'cell_count': 125,
        #     'snr': 0.5,
        #     },
        }

# # This adjusted for the genesnake version. snrs are a bit different scale
# # for this one
# data_cases = {
#         'easy': {
#             'negbin_prob': 0.5,
#             'dispersion': 0.1,
#             'cell_count': 125,
#             'snr': 10,
#             },
#         'low snr': {
#             'negbin_prob': 0.5,
#             'dispersion': 0.1,
#             'cell_count': 125,
#             'snr': 1,
#             },
#         'high dropout': {
#             'negbin_prob': 0.5,
#             'dispersion': 10,
#             'cell_count': 125,
#             'snr': 10,
#             },
#         }

anton_util.pickle_object(data_cases, 'outputs/data_cases.pkl')

repeats = 5
parameter_sets = []
for data_case, parameters in data_cases.items():
    for _ in range(repeats):
        parameter_sets.append(parameters)

simulation_specifications = initialise_simulations(parameter_sets = parameter_sets)
anton_util.pickle_object(
        simulation_specifications,
        'outputs/simulation_specifications.pkl')






