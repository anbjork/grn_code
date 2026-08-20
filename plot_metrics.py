import matplotlib.pyplot as plt
import numpy as np
import os

def plot_metrics_with_jitter(df):
    """
    Plot AUROC, AUPR ratio, and top_k_accuracy with methods on x-axis and jitter.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    metrics = ['AUROC', 'AUPR ratio', 'top_k_accuracy']
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    # Incompatible with constrained_layout
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.5)
    for i, metric in enumerate(metrics):
        ax = axes[i]
        methods = df['method'].unique()
        x_positions = np.arange(len(methods))
        for j, method in enumerate(methods):
            method_data = df[df['method'] == method][metric]
            jitter = np.random.normal(0, 0.1, len(method_data))
            x_jittered = np.full(len(method_data), j) + jitter
            ax.scatter(x_jittered, method_data, alpha=0.7, s=50)
        ax.set_xlabel('Method')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} by Method')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
    # Incompatible with constrained_layout
    # plt.tight_layout()
    fig.suptitle(plot_name)
    return fig

if __name__ == "__main__":

    import anton_util
    from grn_code.pipeline_configuration import output_base_path
    anton_util.log_timestamp('plotting...')
    anton_util.log_timestamp('reading data...')

    df = anton_util.unpickle_object(
            f'{output_base_path}/simulated/compiled_results.pkl')
    output_dir = f'{output_base_path}/simulated/plots'

    df = df.loc[df['inference error'].isna(), :]

    vars_to_stratify = [
            'data_case',
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

    from copy import deepcopy
    # configs = configs[:1]  # Debug
    for config in configs:
        df_subset = deepcopy(df)
        plot_name = ' | '.join([f'{k} {v}' for k, v in config.items()])
        anton_util.log_timestamp(f'{plot_name}...')
        for var_to_stratify, option in config.items():
            df_subset = df_subset[df_subset[var_to_stratify] == option]
        fig = plot_metrics_with_jitter(df_subset)
        fig.savefig(
            f'{output_dir}/{plot_name}.png',
            dpi=300,
            )
        plt.close()

    anton_util.log_timestamp('plotting done')




