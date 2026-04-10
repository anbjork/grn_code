import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_scatter_plot_with_jitter(ax, method_data, metric, color=None, label=None):
    """Create a scatter plot with jitter for better visibility of overlapping points."""
    x_positions = []
    y_values = []
    
    # Custom sorting: numbers first (ascending), then 'False' at the end
    unique_categories = method_data['pseudo_bulk_str'].unique()
    numeric_categories = [cat for cat in unique_categories if cat != 'False']
    # Sort numeric categories as integers
    numeric_categories = sorted(numeric_categories, key=int)
    ordered_categories = numeric_categories + (['False'] if 'False' in unique_categories else [])
    
    for j, pseudo_bulk in enumerate(ordered_categories):
        subset = method_data[method_data['pseudo_bulk_str'] == pseudo_bulk]
        if len(subset) > 1:
            jitter = np.random.normal(0, 0.075, len(subset))
            x_positions.extend([j] * len(subset) + jitter)
        else:
            x_positions.extend([j] * len(subset))
        y_values.extend(subset[metric].values)
    
    ax.scatter(x_positions, y_values, alpha=0.7, s=60, color=color, label=label)
    ax.set_xticks(range(len(ordered_categories)))
    ax.set_xticklabels(ordered_categories)

def create_auroc_plots(df_filtered, methods_to_plot, title_prefix, output_path):
    """Create AUROC comparison plots for all methods."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{title_prefix}: AUROC Performance Comparison across Pseudo-bulk Categories', fontsize=16)

    for i, method in enumerate(methods_to_plot):
        method_data = df_filtered[df_filtered['method'] == method]
        
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        if len(method_data) > 0:
            create_scatter_plot_with_jitter(ax, method_data, 'AUROC')
        
        ax.set_title(f'{method} - AUROC vs Pseudo-bulk')
        ax.set_xlabel('Pseudo-bulk Category')
        ax.set_ylabel('AUROC')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_aupr_ratio_plots(df_filtered, methods_to_plot, title_prefix, output_path):
    """Create AUPR ratio comparison plots for all methods."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{title_prefix}: AUPR Ratio Performance Comparison across Pseudo-bulk Categories', fontsize=16)

    for i, method in enumerate(methods_to_plot):
        method_data = df_filtered[df_filtered['method'] == method]
        
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        if len(method_data) > 0:
            create_scatter_plot_with_jitter(ax, method_data, 'AUPR ratio')
        
        ax.set_title(f'{method} - AUPR Ratio vs Pseudo-bulk')
        ax.set_xlabel('Pseudo-bulk Category')
        ax.set_ylabel('AUPR Ratio')
        ax.set_ylim(0, None)  # Start from 0, let matplotlib determine upper limit
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_aupr_erma_plots(df_filtered, methods_to_plot, title_prefix, output_path):
    """Create AUPR and ERMA comparison plots for all methods."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{title_prefix}: AUPR and ERMA Performance Comparison across Pseudo-bulk Categories', fontsize=16)

    for i, method in enumerate(methods_to_plot):
        method_data = df_filtered[df_filtered['method'] == method]
        
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        if len(method_data) > 0:
            # Custom sorting: numbers first (ascending), then 'False' at the end
            unique_categories = method_data['pseudo_bulk_str'].unique()
            numeric_categories = [cat for cat in unique_categories if cat != 'False']
            # Sort numeric categories as integers
            numeric_categories = sorted(numeric_categories, key=int)
            ordered_categories = numeric_categories + (['False'] if 'False' in unique_categories else [])
            
            # Add jitter to x-axis for better visibility (only when multiple points)
            for j, pseudo_bulk in enumerate(ordered_categories):
                subset = method_data[method_data['pseudo_bulk_str'] == pseudo_bulk]
                if len(subset) > 1:
                    jitter = np.random.normal(0, 0.075, len(subset))
                    x_pos = [j] * len(subset) + jitter
                else:
                    x_pos = [j] * len(subset)
                
                ax.scatter(x_pos, subset['AUPR'].values, 
                           alpha=0.7, s=60, label='AUPR' if j == 0 else "", color='blue')
                ax.scatter(x_pos, subset['ERMA'].values, 
                           alpha=0.7, s=60, label='ERMA' if j == 0 else "", color='red')
            
            aupr_max = method_data['AUPR'].max()
            erma_max = method_data['ERMA'].max()
            y_max = max(aupr_max, erma_max) * 1.1
            ax.set_ylim(0, y_max)
            ax.set_xticks(range(len(ordered_categories)))
            ax.set_xticklabels(ordered_categories)
            ax.legend()
        
        ax.set_title(f'{method} - AUPR and ERMA vs Pseudo-bulk')
        ax.set_xlabel('Pseudo-bulk Category')
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
