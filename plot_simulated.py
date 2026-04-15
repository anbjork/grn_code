import anton_util
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from plot_functions import create_auroc_plots, create_aupr_erma_plots, create_aupr_ratio_plots

# Create output directory
os.makedirs('./plots/simulated', exist_ok=True)

# Load the data
df = anton_util.unpickle_object('benchmarks/simulated/stats_df.pkl')

# Filter for non-shuffled data and specific methods
methods_to_plot = ['lsco', 'lsco.T', 'zscore_ab', 'zscore_dream3']
df_filtered = df[(df['method'].isin(methods_to_plot)) & (df['is_shuffled'] == False)].copy()

# Convert pseudo_bulk to string for better plotting
df_filtered['pseudo_bulk_str'] = df_filtered['pseudo_bulk'].astype(str)

# Check what's actually in the delta column
print(f"Delta column unique values: {df_filtered['delta'].unique()}")
print(f"Delta column value counts:\n{df_filtered['delta'].value_counts(dropna=False)}")

# Handle delta values properly - convert NaN to 'False' and True to 'True'
df_filtered['delta_clean'] = df_filtered['delta'].apply(lambda x: 'False' if pd.isna(x) else 'True')
unique_deltas = df_filtered['delta_clean'].unique()
print(f"Unique deltas after cleaning: {unique_deltas}")

# Get unique transforms for color mapping
unique_transforms = df_filtered['transform'].unique()
print(f"Unique transforms found: {unique_transforms}")

# Get unique replicates for color mapping
unique_replicates = df_filtered['replicate'].unique()
print(f"Unique replicates found: {unique_replicates}")

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

print(f"Total rows after filtering: {len(df_filtered)}")
print("Data points per method and pseudo_bulk category:")
for method in methods_to_plot:
    method_data = df_filtered[df_filtered['method'] == method]
    print(f"\n{method}:")
    counts = method_data['pseudo_bulk'].value_counts().sort_index()
    print(counts)

# Create plots colored by delta
create_auroc_plots(df_filtered, methods_to_plot, 'Simulated Data', './plots/simulated/auroc_comparison_scatter_delta.png', color_by='delta')
create_aupr_erma_plots(df_filtered, methods_to_plot, 'Simulated Data', './plots/simulated/aupr_erma_comparison_scatter_delta.png', color_by='delta')
create_aupr_ratio_plots(df_filtered, methods_to_plot, 'Simulated Data', './plots/simulated/aupr_ratio_comparison_scatter_delta.png', color_by='delta')

# Create plots colored by transform
create_auroc_plots(df_filtered, methods_to_plot, 'Simulated Data', './plots/simulated/auroc_comparison_scatter_transform.png', color_by='transform')
create_aupr_erma_plots(df_filtered, methods_to_plot, 'Simulated Data', './plots/simulated/aupr_erma_comparison_scatter_transform.png', color_by='transform')
create_aupr_ratio_plots(df_filtered, methods_to_plot, 'Simulated Data', './plots/simulated/aupr_ratio_comparison_scatter_transform.png', color_by='transform')

# Create plots colored by replicate
create_auroc_plots(df_filtered, methods_to_plot, 'Simulated Data', './plots/simulated/auroc_comparison_scatter_replicate.png', color_by='replicate')
create_aupr_erma_plots(df_filtered, methods_to_plot, 'Simulated Data', './plots/simulated/aupr_erma_comparison_scatter_replicate.png', color_by='replicate')
create_aupr_ratio_plots(df_filtered, methods_to_plot, 'Simulated Data', './plots/simulated/aupr_ratio_comparison_scatter_replicate.png', color_by='replicate')

