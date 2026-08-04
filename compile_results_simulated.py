
import pandas as pd
import anton_util

anton_util.log_timestamp('compiling results...')

data_sources, inferred, benchmarks = [
    anton_util.unpickle_object(f'outputs/simulated/{name}.pkl')
    for name in ['data_processed', 'inferences', 'benchmarks']
    ]

results = []
for benchmark in benchmarks:
    tmp = {}
    for d in benchmark.values():
        try:
            tmp.update(d)
        except TypeError as e:
            print(e)
            print('If benchmarking fails, the data is None, so could be that')
    results.append(tmp)



output_path = 'outputs/simulated/compiled_results'
df = pd.DataFrame(results)
cs = ['shuffle', 'method', 'pseudo_bulk']
df = df.sort_values(by = cs)
df['AUPR ratio'] = df['AUPR'] / df['ERMA']
anton_util.pickle_object(df, f'{output_path}.pkl')
# csvs don't handle the nested f1_scores well
drop_cols = ['f1_scores', 'plot_roc', 'plot_pr']
df = df.drop(drop_cols, axis = 1)
df.to_csv(f'{output_path}.csv')

anton_util.log_timestamp('compilation done')


