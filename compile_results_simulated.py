
import pandas as pd
from pathlib import Path
import anton_util



path = Path('data_processed/simulated/preprocessed.pkl')
data_sources = anton_util.unpickle_object(path)

inference_dir = Path('inferences/simulated')
inferred = anton_util.unpickle_object(
    inference_dir / 'estimated_networks.pkl')

benchmark_output_dir = Path('benchmarks/simulated')
benchmarks = anton_util.unpickle_object(
    benchmark_output_dir / 'stats.pkl')

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




df = pd.DataFrame(results)
cs = ['shuffle', 'method', 'pseudo_bulk']
if all([elem in df.columns for elem in cs]):
    df['is_shuffled'] = [elem is not False for elem in df['shuffle']]
    df = df.sort_values(by = ['is_shuffled', 'method', 'pseudo_bulk'])
df['AUPR ratio'] = df['AUPR'] / df['ERMA']
anton_util.pickle_object(df, benchmark_output_dir / 'stats_df.pkl')
# csvs don't handle the nested f1_scores well
df = df.drop('f1_scores', axis = 1)
df.to_csv(benchmark_output_dir / 'stats_df.csv')

