
import pandas as pd
from pathlib import Path
import anton_util

benchmark_output_dir = Path('benchmarks/replogle')
benchmarks = anton_util.unpickle_object(
    benchmark_output_dir / 'stats.pkl')

all = []
for benchmark in benchmarks:
    tmp = {}
    for d in benchmark.values():
        tmp.update(d)
    all.append(tmp)




anton_util.pickle_object(all, benchmark_output_dir / 'compiled_stats.pkl')
df = pd.DataFrame(all)
# df['is_shuffled'] = [elem is not False for elem in df.shuffle]
# df = df.sort_values(by = ['is_shuffled', 'method', 'pseudo_bulk'])
df = df.sort_values(by = ['method', 'pseudo_bulk'])
df['AUPR ratio'] = df['AUPR'] / df['ERMA']
anton_util.pickle_object(df, benchmark_output_dir / 'stats_df.pkl')
# csvs don't handle the nested f1_scores well
df = df.drop('f1_scores', axis = 1)
df.to_csv(benchmark_output_dir / 'stats_df.csv')





