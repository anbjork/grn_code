
import pandas as pd
from pathlib import Path
import anton_util


tmp = Path('data/replogle')
tmp.mkdir(exist_ok=True, parents=True)
data_set_name = 'K562_essential_raw_singlecell_01'
anton_util.log_timestamp(f'Loading {data_set_name}...')
data_sources = anton_util.unpickle_object(
    f'{tmp}/{data_set_name}_preprocessed_2.pkl')
anton_util.log_timestamp(f'{data_set_name} loaded.')

inference_dir = Path('inferences/replogle')
inferred = anton_util.unpickle_object(
    inference_dir / 'estimated_networks.pkl')

benchmark_output_dir = Path('benchmarks/replogle')
benchmarks = anton_util.unpickle_object(
    benchmark_output_dir / 'stats.pkl')

all = []
for data_source, benchmark in zip(data_sources, benchmarks):
    for method, mstats in benchmark.items():
        for ref_name, stats in mstats.items():
            all.append({
                'shuffle': data_source['shuffle'],
                'method': method,
                # This one already included in stats
                # 'reference': ref_name,
                **stats,
                })

anton_util.pickle_object(all, benchmark_output_dir / 'compiled_stats.pkl')
df = pd.DataFrame(all)
df.to_csv(benchmark_output_dir / 'stats_df.csv')
df['is_shuffled'] = [elem is not False for elem in df.shuffle]
anton_util.pickle_object(df, benchmark_output_dir / 'stats_df.pkl')
dfd = df.drop('f1_scores', axis = 1)
dfd = dfd.sort_values(by = ['is_shuffled', 'method'])




