
import pandas as pd
from pathlib import Path
import anton_util



path = Path('data/simulated/preprocessed.pkl')
data_sources = anton_util.unpickle_object(path)

inference_dir = Path('inferences/simulated')
inferred = anton_util.unpickle_object(
    inference_dir / 'estimated_networks.pkl')

benchmark_output_dir = Path('benchmarks/simulated')
benchmarks = anton_util.unpickle_object(
    benchmark_output_dir / 'stats.pkl')

all = []
for data_source, estimated_networks, stats in zip(data_sources, inferred, benchmarks):
    for method, estimated_network in estimated_networks.items():
        all.append({
            'method': method,
            'dataset': data_source['index'],
            'shuffle': data_source['shuffle'],
            'pseudo_bulk': data_source['pseudo_bulk'],
            **stats[method],
        })

df = pd.DataFrame(all)
df['is_shuffled'] = [elem is not False for elem in df.shuffle]
df = df.sort_values(by = ['is_shuffled', 'method', 'pseudo_bulk'])
df['AUPR ratio'] = df['AUPR'] / df['ERMA']
anton_util.pickle_object(df, benchmark_output_dir / 'stats_df.pkl')
# csvs don't handle the nested f1_scores well
df = df.drop('f1_scores', axis = 1)
df.to_csv(benchmark_output_dir / 'stats_df.csv')

