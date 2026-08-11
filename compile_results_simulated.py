
import pandas as pd
import anton_util


anton_util.log_timestamp('reading data...')
from grn_code.pipeline_configuration import output_base_path
benchmarks = anton_util.unpickle_object(f'{output_base_path}/simulated/benchmarks.pkl')
# data_sources, inferred, benchmarks = [
        # broken on purpose, since not updated and tested for
        # import of paths from pipeline configuration
#     anton_util.unpickle_object(f'outputs/simulated/{name}.pkl')
#     for name in ['data_processed', 'inferences', 'benchmarks']
#     ]


anton_util.log_timestamp('compiling results...')

def recursive_flatten_dict(dd):
    flat = {}
    for k, v in dd.items():
        try:
            add = recursive_flatten_dict(v)
            flat.update(add)
        except AttributeError:
            flat[k] = v
    return flat

results = []
for benchmark in benchmarks:
    tmp = {}
    from copy import deepcopy
    benchmark = deepcopy(benchmark)
    benchmark['data'].pop('f1_scores')
    tmp = recursive_flatten_dict(benchmark)
    results.append(tmp)

output_path = f'{output_base_path}/simulated/compiled_results'
df = pd.DataFrame(results)
cs = ['shuffle', 'method', 'pseudo_bulk']
df = df.sort_values(by = cs)
df['AUPR ratio'] = df['AUPR'] / df['ERMA']

anton_util.log_timestamp('saving results...')

anton_util.pickle_object(df, f'{output_path}.pkl')
drop_cols = ['plot_roc', 'plot_pr']
df = df.drop(drop_cols, axis = 1)
df.to_csv(f'{output_path}.csv')

anton_util.log_timestamp('compilation done')


