
import anton_util

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

aa = anton_util.unpickle_object('compiled_results.pkl')

data_cases = ['high dropout']
methods = ['lsco.T', 'zscore_ab']
plot_type = ['plot_roc', 'plot_pr']

for data_case in data_cases:
    for method in methods:
        for plot in plot_type:
            bb = aa.loc[(aa.data_case == data_case) & (aa.method == method)]
            print(bb)
            for _, row in bb.iterrows():
                row[plot].savefig(f'{plot} | {method} | {data_case} | replicate {row.replicate}.svg')

# bb = aa.loc[(aa.data_case == 'high dropout') & (aa.method == 'lsco.T')]
# for _, row in bb.iterrows():
#     row.plot_roc.savefig(f'lsco.T | high dropout | replicate {row.replicate}.svg')



