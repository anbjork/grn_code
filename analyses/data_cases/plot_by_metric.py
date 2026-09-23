import matplotlib.pyplot as plt
import numpy as np
import os


def plot_metrics(output_path):

    def plot_metric_across_data_cases(df, metric):
        """
        One plot per metric, one subplot per data case.
        Methods are on the x-axis within each subplot.
        """
        os.makedirs(output_dir, exist_ok=True)
        data_cases = df['data_case'].unique()
        methods = df['method'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(methods)))
        x_positions = np.arange(len(methods))

        fig, axes = plt.subplots(1, len(data_cases),
                                 figsize=(5 * len(data_cases), 4.5),
                                 sharey=True)
        if len(data_cases) == 1:
            axes = [axes]
        plt.subplots_adjust(bottom=0.4)

        for i, (ax, data_case) in enumerate(zip(axes, data_cases)):
            df_case = df[df['data_case'] == data_case]
            for j, method in enumerate(methods):
                method_data = df_case[df_case['method'] == method][metric]
                jitter = np.random.normal(0, 0.1, len(method_data))
                ax.scatter(x_positions[j] + jitter, method_data,
                           alpha=0.7, s=50, color=colors[j])
            ax.set_xticks(x_positions)
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.set_title(data_case)
            ax.set_ylim(None, y_maxes[metric] * 1.1)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_ylabel(metric)

        fig.suptitle(metric)
        return fig


    import anton_util
    anton_util.log_timestamp('plotting...')
    anton_util.log_timestamp('reading data...')

    df = anton_util.unpickle_object(
            f'{output_path}/simulated/compiled_results.pkl')
    output_dir = f'{output_path}/simulated/plots'

    df = df.loc[df['inference error'].isna(), :]

    vars_to_stratify = [
            'cell normalised',
            'read normalised',
            'transform 1',
            'transform 2', 
            'pseudo_bulk',
            ]
    options = {v: df[v].unique() for v in vars_to_stratify}
    configs = []
    def recursive_combos(determined, remaining):
        from copy import deepcopy
        if len(remaining) == 0:
            configs.append(deepcopy(determined))
            return
        k, options = remaining.popitem()
        for option in options:
            determined[k] = option
            recursive_combos(determined, deepcopy(remaining))
    recursive_combos({}, options)
    # print('Configs to plot:')
    # from pprint import pprint
    # pprint(configs)


    metrics = ['AUROC', 'AUPR ratio', 'top_k_accuracy']
    y_maxes = {}
    for metric in metrics:
        y_maxes[metric] = df[metric].max()

    from copy import deepcopy
    # configs = configs[:1]  # Debug
    for config in configs:
        df_subset = deepcopy(df)
        plot_name = ' | '.join([f'{k} {v}' for k, v in config.items()])
        anton_util.log_timestamp(f'{plot_name}...')
        for var_to_stratify, option in config.items():
            df_subset = df_subset[df_subset[var_to_stratify] == option]
        for metric in metrics:
            anton_util.log_timestamp(f'  {metric}...')
            fig = plot_metric_across_data_cases(df_subset, metric)
            fig.savefig(
                f'{output_dir}/{metric} | {plot_name}.png',
                dpi=300,
                bbox_inches='tight',
                )
            plt.close()

    anton_util.log_timestamp('plotting done')




