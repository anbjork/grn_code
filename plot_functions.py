import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_color_mapping(values, color_by):
    """Create a color mapping for values."""
    unique_values = sorted(list(set(values)))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_values)))
    return dict(zip(unique_values, colors))

def create_scatter_plot_with_jitter(ax, method_data, metric, color_map=None, color_by='delta', show_legend=False):
    """Create a scatter plot with jitter for better visibility of overlapping points."""
    # Custom sorting: numbers first (ascending), then 'False' at the end
    unique_categories = method_data['pseudo_bulk_str'].unique()
    numeric_categories = [cat for cat in unique_categories if cat != 'False']
    # Sort numeric categories as integers
    numeric_categories = sorted(numeric_categories, key=int)
    ordered_categories = numeric_categories + (['False'] if 'False' in unique_categories else [])
    
    # Track which values we've seen for legend - move outside the loop
    legend_values = set()
    
    # Determine which column to use for coloring
    color_column = None
    if color_by is not None:
        if color_by == 'delta':
            color_column = 'delta_clean'
        elif color_by == 'transform':
            color_column = 'transform'
        else:  # replicate
            color_column = 'replicate'
    
    for j, pseudo_bulk in enumerate(ordered_categories):
        subset = method_data[method_data['pseudo_bulk_str'] == pseudo_bulk]
        if len(subset) > 1:
            jitter = np.random.normal(0, 0.075, len(subset))
            x_positions = [j] * len(subset) + jitter
        else:
            x_positions = [j] * len(subset)
        
        # Color by specified column if color_map provided and column exists
        if color_map is not None and color_column is not None and color_column in method_data.columns:
            for value in subset[color_column].unique():
                value_subset = subset[subset[color_column] == value]
                if len(value_subset) > 1:
                    value_jitter = np.random.normal(0, 0.075, len(value_subset))
                    value_x_positions = [j] * len(value_subset) + value_jitter
                else:
                    value_x_positions = [j] * len(value_subset)
                
                # Only add label if we haven't seen this value before AND we want to show legend
                label = str(value) if (show_legend and value not in legend_values) else None
                if label:
                    legend_values.add(value)
                
                ax.scatter(value_x_positions, value_subset[metric].values, 
                          alpha=0.7, s=60, color=color_map[value], label=label)
        else:
            ax.scatter(x_positions, subset[metric].values, alpha=0.7, s=60)
    
    ax.set_xticks(range(len(ordered_categories)))
    ax.set_xticklabels(ordered_categories)

def create_auroc_plots(df_filtered, methods_to_plot, title_prefix, output_path, color_by='delta'):
    """Create AUROC comparison plots for all methods."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Handle coloring
    color_map = None
    legend_title = None
    if color_by is not None:
        # Set title and color mapping based on color_by parameter
        if color_by == 'delta':
            color_column = 'delta_clean'
        elif color_by == 'transform':
            color_column = 'transform'
        else:  # replicate
            color_column = 'replicate'
        
        # Check if the column exists
        if color_column in df_filtered.columns:
            color_values = df_filtered[color_column]
            color_map = get_color_mapping(color_values, color_by)
            legend_title = color_by.capitalize()
            fig.suptitle(f'{title_prefix}: AUROC Performance Comparison across Pseudo-bulk Categories (Colored by {legend_title})', fontsize=16)
        else:
            fig.suptitle(f'{title_prefix}: AUROC Performance Comparison across Pseudo-bulk Categories', fontsize=16)
    else:
        fig.suptitle(f'{title_prefix}: AUROC Performance Comparison across Pseudo-bulk Categories', fontsize=16)

    for i, method in enumerate(methods_to_plot):
        method_data = df_filtered[df_filtered['method'] == method]
        
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        if len(method_data) > 0:
            create_scatter_plot_with_jitter(ax, method_data, 'AUROC', color_map, color_by, show_legend=(i==0))
        
        ax.set_title(f'{method} - AUROC vs Pseudo-bulk')
        ax.set_xlabel('Pseudo-bulk Category')
        ax.set_ylabel('AUROC')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add legend only to first subplot if we have coloring
        if i == 0 and color_map is not None and legend_title is not None:
            ax.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_aupr_ratio_plots(df_filtered, methods_to_plot, title_prefix, output_path, color_by='delta'):
    """Create AUPR ratio comparison plots for all methods."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Handle coloring
    color_map = None
    legend_title = None
    if color_by is not None:
        # Set title and color mapping based on color_by parameter
        if color_by == 'delta':
            color_column = 'delta_clean'
        elif color_by == 'transform':
            color_column = 'transform'
        else:  # replicate
            color_column = 'replicate'
        
        # Check if the column exists
        if color_column in df_filtered.columns:
            color_values = df_filtered[color_column]
            color_map = get_color_mapping(color_values, color_by)
            legend_title = color_by.capitalize()
            fig.suptitle(f'{title_prefix}: AUPR Ratio Performance Comparison across Pseudo-bulk Categories (Colored by {legend_title})', fontsize=16)
        else:
            fig.suptitle(f'{title_prefix}: AUPR Ratio Performance Comparison across Pseudo-bulk Categories', fontsize=16)
    else:
        fig.suptitle(f'{title_prefix}: AUPR Ratio Performance Comparison across Pseudo-bulk Categories', fontsize=16)

    for i, method in enumerate(methods_to_plot):
        method_data = df_filtered[df_filtered['method'] == method]
        
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        if len(method_data) > 0:
            create_scatter_plot_with_jitter(ax, method_data, 'AUPR ratio', color_map, color_by, show_legend=(i==0))
        
        ax.set_title(f'{method} - AUPR Ratio vs Pseudo-bulk')
        ax.set_xlabel('Pseudo-bulk Category')
        ax.set_ylabel('AUPR Ratio')
        ax.set_ylim(0, None)  # Start from 0, let matplotlib determine upper limit
        ax.grid(True, alpha=0.3)
        
        # Add legend only to first subplot if we have coloring
        if i == 0 and color_map is not None and legend_title is not None:
            ax.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_aupr_erma_plots(df_filtered, methods_to_plot, title_prefix, output_path, color_by='delta'):
    """Create AUPR and ERMA comparison plots for all methods."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Handle coloring
    color_map = None
    legend_title = None
    color_column = None
    if color_by is not None:
        # Set title and color mapping based on color_by parameter
        if color_by == 'delta':
            color_column = 'delta_clean'
        elif color_by == 'transform':
            color_column = 'transform'
        else:  # replicate
            color_column = 'replicate'
        
        # Check if the column exists
        if color_column in df_filtered.columns:
            color_values = df_filtered[color_column]
            color_map = get_color_mapping(color_values, color_by)
            legend_title = color_by.capitalize()
            fig.suptitle(f'{title_prefix}: AUPR and ERMA Performance Comparison across Pseudo-bulk Categories (Colored by {legend_title})', fontsize=16)
        else:
            fig.suptitle(f'{title_prefix}: AUPR and ERMA Performance Comparison across Pseudo-bulk Categories', fontsize=16)
    else:
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
            
            # Track legend entries
            legend_entries = set()
            
            # Add jitter to x-axis for better visibility (only when multiple points)
            for j, pseudo_bulk in enumerate(ordered_categories):
                subset = method_data[method_data['pseudo_bulk_str'] == pseudo_bulk]
                
                if color_map is not None and color_column is not None:
                    # Plot AUPR points colored by specified column
                    for value in subset[color_column].unique():
                        value_subset = subset[subset[color_column] == value]
                        if len(value_subset) > 1:
                            jitter = np.random.normal(0, 0.075, len(value_subset))
                            x_pos = [j] * len(value_subset) + jitter
                        else:
                            x_pos = [j] * len(value_subset)
                        
                        # AUPR points (circles)
                        aupr_label = f'{value} (AUPR)' if (i == 0 and f'{value}_AUPR' not in legend_entries) else ""
                        if aupr_label:
                            legend_entries.add(f'{value}_AUPR')
                        ax.scatter(x_pos, value_subset['AUPR'].values, 
                                   alpha=0.7, s=60, color=color_map[value], 
                                   marker='o', label=aupr_label)
                        
                        # ERMA points (triangles)
                        erma_label = f'{value} (ERMA)' if (i == 0 and f'{value}_ERMA' not in legend_entries) else ""
                        if erma_label:
                            legend_entries.add(f'{value}_ERMA')
                        ax.scatter(x_pos, value_subset['ERMA'].values, 
                                   alpha=0.7, s=60, color=color_map[value], 
                                   marker='^', label=erma_label)
                else:
                    # No coloring - plot all points with default colors
                    if len(subset) > 1:
                        jitter = np.random.normal(0, 0.075, len(subset))
                        x_pos = [j] * len(subset) + jitter
                    else:
                        x_pos = [j] * len(subset)
                    
                    # AUPR points (circles)
                    aupr_label = 'AUPR' if (i == 0 and j == 0) else ""
                    ax.scatter(x_pos, subset['AUPR'].values, 
                               alpha=0.7, s=60, color='blue', 
                               marker='o', label=aupr_label)
                    
                    # ERMA points (triangles)
                    erma_label = 'ERMA' if (i == 0 and j == 0) else ""
                    ax.scatter(x_pos, subset['ERMA'].values, 
                               alpha=0.7, s=60, color='red', 
                               marker='^', label=erma_label)
            
            aupr_max = method_data['AUPR'].max()
            erma_max = method_data['ERMA'].max()
            y_max = max(aupr_max, erma_max) * 1.1
            ax.set_ylim(0, y_max)
            ax.set_xticks(range(len(ordered_categories)))
            ax.set_xticklabels(ordered_categories)
            
            # Add legend only to first subplot
            if i == 0:
                if color_map is not None and legend_title is not None:
                    ax.legend(title=f'{legend_title} & Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    ax.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax.set_title(f'{method} - AUPR and ERMA vs Pseudo-bulk')
        ax.set_xlabel('Pseudo-bulk Category')
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
