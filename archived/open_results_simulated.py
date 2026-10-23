
import anton_util
import pandas as pd


df = anton_util.unpickle_object('outputs/simulated/compiled_results.pkl')


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# df = df.drop('f1_scores', axis = 1)
df = df.rename({'n_genes_after_harmonisation': 'n_genes'}, axis = 1)

# cs = ['is_shuffled', 'method', 'transform', 'delta']
cs = ['method', 'transform']
if all([elem in df.columns for elem in cs]):
    df = df.sort_values(by = cs)
dfd = df.reset_index(drop = True)

# print(dfd)


if False:
    # Checking dropouts
    import numpy as np
    tmp = [True if 'fraction' in name else False for name in dfd.columns]
    tmp = (np.array(tmp) | (dfd.columns == 'data_case'))
    aa = dfd.loc[:, tmp]
    means = {}
    for gn, g in aa.groupby('data_case'):
        means[gn] = g.drop('data_case', axis = 1).mean()
    dfm = pd.DataFrame(means).T


