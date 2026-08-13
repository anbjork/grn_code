
import pandas as pd
import anton_util
from copy import deepcopy

anton_util.log_timestamp('compiling results...')

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
    # anton_util.log_timestamp(f'copying')
    # Performance optimisation. Sacrifices f1_scores from outputs
    # in the pickle below, but goes from 20 s to negligible
    # benchmark = deepcopy(benchmark)
    # anton_util.log_timestamp(f'popping')
    benchmark['data'].pop('f1_scores')
    # anton_util.log_timestamp(f'flattening benchmark')
    tmp = recursive_flatten_dict(benchmark)
    # anton_util.log_timestamp(f'the rest')
    results.append(tmp)

output_path = f'{output_base_path}/simulated/compiled_results'
df = pd.DataFrame(results)
cs = ['shuffle', 'method', 'pseudo_bulk']
df = df.sort_values(by = cs)
df['AUPR ratio'] = df['AUPR'] / df['ERMA']






CONFIG_plot_roc_and_pr = True

if CONFIG_plot_roc_and_pr:

    anton_util.log_timestamp('plotting rocs and prs...')

    plot_out_dir = f'{output_base_path}/simulated/plots/roc_and_pr/'
    from pathlib import Path
    Path(plot_out_dir).mkdir(parents=True, exist_ok=True)

    parameters = []
    for benchmark in benchmarks:
        p = deepcopy(benchmark['meta'])
        for k in ['0_fraction', 'any nan', 'inference error']:
            p.pop(k)
        parameters.append(p)
    parameters_flat = [recursive_flatten_dict(p) for p in parameters]


    for ps, rs in zip(parameters_flat, results):
        for plot_type in ['plot_roc', 'plot_pr']:
            ps['plot_type'] = plot_type
            tag = ' / '.join([f'{k} {v}' for k, v in ps.items()])
            tag = f'plot_type {plot_type} / ' + tag
            filename = f'{output_base_path}/simulated/plots/roc_and_pr/{tag}.svg'
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            # Here, the actual savefig command is taking all the time
            # So no reason to optimise anything else
            # anton_util.log_timestamp(f'saving plot')
            rs[plot_type].savefig(filename)
            # anton_util.log_timestamp(f'the rest')







anton_util.log_timestamp('saving results...')
anton_util.pickle_object(df, f'{output_path}.pkl')
drop_cols = ['plot_roc', 'plot_pr']
df = df.drop(drop_cols, axis = 1)
df.to_csv(f'{output_path}.csv')

anton_util.log_timestamp('compilation done')


