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

# AUROC plot
create_auroc_plots(df_filtered, methods_to_plot, 'Simulated Data', './plots/simulated/auroc_comparison_scatter.png')

# AUPR and ERMA plot
create_aupr_erma_plots(df_filtered, methods_to_plot, 'Simulated Data', './plots/simulated/aupr_erma_comparison_scatter.png')

# AUPR ratio plot
create_aupr_ratio_plots(df_filtered, methods_to_plot, 'Simulated Data', './plots/simulated/aupr_ratio_comparison_scatter.png')

# Print summary statistics
print("\nSummary Statistics by Method and Pseudo-bulk Category:")
print("=" * 60)
for method in methods_to_plot:
    method_data = df_filtered[df_filtered['method'] == method]
    print(f"\n{method.upper()}:")
    summary = method_data.groupby('pseudo_bulk_str')[['AUROC', 'AUPR', 'ERMA', 'AUPR ratio']].agg(['mean', 'std'])
    print(summary.round(4))
