
import numpy as np
import anton_util
import matplotlib.pyplot as plt
from local_imports import output_path

anton_util.log_timestamp('plotting...')
anton_util.log_timestamp('reading data...')

df = anton_util.unpickle_object(
        f'{output_path}/simulated/compiled_results.pkl')

anton_util.log_timestamp('the rest...')

from pathlib import Path
output_dir = Path(f'{output_path}/simulated/plots')
output_dir.mkdir(exist_ok=True, parents=True)

dropout_col = '0_fraction__before_filtering__all'
x_vars = ['dispersion', 'snr', 'cell_count']
y_var = 'AUROC'

controlled_vars = {'snr', 'cell_count'}
dependent_vars = {'dispersion': dropout_col}
xlims = {
        'snr': (None, 0.1),
        }

df['controlled_var'] = [elem.split(':')[0] for elem in df.data_case]

for controlled_var in x_vars:
    dfs = df[df['controlled_var'] == controlled_var]
    # dfs = df  # Debug
    if controlled_var in dependent_vars.keys():
        x_var = dependent_vars[controlled_var]
    else:
        x_var = controlled_var
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = sorted(set(dfs['method']))
    colors = plt.get_cmap('tab20').colors  # pyright: ignore
    for i, method in enumerate(methods):
        df_method = dfs[dfs['method'] == method]
        df_method = df_method.sort_values(by=x_var)

        if x_var in controlled_vars:
            grouped = df_method.groupby(x_var)[y_var]
            means = grouped.mean()
            sds = grouped.std()
            ax.errorbar(means.index, means, yerr=sds, label=method, capsize=3, color=colors[i % len(colors)])
        else:
            x = np.array(df_method[x_var])
            smooth_auroc = np.array(df_method[y_var].rolling(window=10, center=True).mean())
            ax.plot(x, smooth_auroc, label=method, color=colors[i % len(colors)])
    # ax.set_xlim(0.75, 1)
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0)
    plt.tight_layout()

    fig.savefig(f'{output_dir}/auroc_against_{x_var}.png')
    if x_var in xlims.keys():
        ax.set_xlim(xlims[x_var])  # pyright: ignore
        fig.savefig(f'{output_dir}/auroc_against_{x_var}__xlims.png')

anton_util.log_timestamp('done')


plt.close('all')
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['dispersion'], df[dropout_col])
fig.savefig(f'{output_dir}/dropouts_against_dispersion.png')







