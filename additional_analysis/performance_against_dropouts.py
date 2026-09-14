
import anton_util
from grn_code.pipeline_configuration import output_base_path
anton_util.log_timestamp('plotting...')
anton_util.log_timestamp('reading data...')

df = anton_util.unpickle_object(
        f'{output_base_path}/simulated/compiled_results.pkl')

anton_util.log_timestamp('the rest...')

output_dir = f'{output_base_path}/simulated/plots'

c1 = ['high dropout' in elem for elem in df['data_case']]
import numpy as np
c2 = np.array(df['data_case'] == 'easy')
dfs = df[c1 | c2]


import matplotlib.pyplot as plt
plt.close('all')
fig, ax = plt.subplots(figsize=(10, 6))

x_var = '0_fraction__before_filtering__all'
y_var = 'AUROC'
methods = sorted(set(dfs['method']))
for method in methods:
    df_method = dfs[dfs['method'] == method]
    df_method = df_method.sort_values(by=x_var)

    x = np.array(df_method[x_var])
    smooth_auroc = np.array(df_method[y_var].rolling(window=15, center=True).mean())
    # smooth_auroc = df_method[y_var]

    ax.plot(x, smooth_auroc, label=method)
# ax.set_xlim(0.75, 1)
ax.set_xlabel(x_var)
ax.set_ylabel(y_var)
plt.legend(bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0)
plt.tight_layout()

fig.savefig(f'{output_dir}/auroc_against_dropouts.png')

anton_util.log_timestamp('done')


plt.close('all')
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(dfs['dispersion'], dfs[x_var])
fig.savefig(f'{output_dir}/dropouts_against_dispersion.png')
# Soo, the gap was mainly between dispersion 5 and 10, not 0.1 and 10,
# like I assumed. Not an unreasonable assumption. Murphys law though haha







