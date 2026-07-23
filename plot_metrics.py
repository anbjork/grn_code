import matplotlib.pyplot as plt
import numpy as np
import os

def plot_metrics_with_jitter(df, output_prefix='metrics_plot'):
    """
    Plot AUROC, AUPR ratio, and top_k_accuracy with methods on x-axis and jitter.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ['AUROC', 'AUPR ratio', 'top_k_accuracy']
    
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Get unique methods
        methods = df['method'].unique()
        
        # Create x positions for methods
        x_positions = np.arange(len(methods))
        
        # Plot each method with jitter
        for j, method in enumerate(methods):
            method_data = df[df['method'] == method][metric]
            
            # Add random jitter to x position
            jitter = np.random.normal(0, 0.1, len(method_data))
            x_jittered = np.full(len(method_data), j) + jitter
            
            ax.scatter(x_jittered, method_data, alpha=0.7, s=50)
        
        # Customize plot
        ax.set_xlabel('Method')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} by Method')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{output_prefix}.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":

    import anton_util
    anton_util.log_timestamp('plotting...')
    df = anton_util.unpickle_object('outputs/simulated/compiled_results.pkl')
    output_dir = 'outputs/simulated/plots'
    plot_metrics_with_jitter(df)
    print(f"Plots saved to {output_dir}")





