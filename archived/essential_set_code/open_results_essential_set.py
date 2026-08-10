
import anton_util
import pandas as pd


df = anton_util.unpickle_object('benchmarks/replogle/stats_df.pkl')


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

df = df.drop('f1_scores', axis = 1)
df = df.rename({'n_genes_after_harmonisation': 'n_genes'}, axis = 1)

# df = df.sort_values(by = ['transform', 'delta', 'method'])
df = df.sort_values(by = ['transform', 'method'])

dfd = df.reset_index(drop = True)

print(dfd)

