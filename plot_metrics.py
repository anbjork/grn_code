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
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
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
    plt.tight_layout()
    return fig

if __name__ == "__main__":

    import anton_util
    anton_util.log_timestamp('plotting...')
    df = anton_util.unpickle_object('outputs/simulated/compiled_results.pkl')
    output_dir = 'outputs/simulated/plots'

    vars_to_stratify = [
            'pseudo_bulk',
            'cell normalised',
            'pseudo_bulk',
            'read normalised',
            'transform 2', 
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
    print('Configs to plot:')
    from pprint import pprint
    pprint(configs)

    from copy import deepcopy
    for config in configs:
        df_subset = deepcopy(df)
        plot_name = ' | '.join([f'{k} {v}' for k, v in config.items()])
        for var_to_stratify, option in config.items():
            df_subset = df_subset[df_subset[var_to_stratify] == option]
        fig = plot_metrics_with_jitter(df_subset)
        fig.savefig(
            f'{output_dir}/{plot_name}.png',
            dpi=300,
            # bbox_inches='tight'
            )
        plt.close()





