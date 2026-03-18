
from pathlib import Path
import numpy as np
import pandas as pd
import genesnake as gs
import anton_util

import functions

anton_util.log_timestamp('script started')


tmp = Path('beeline_data')
refnets_base_dir = tmp / 'inputs/TF+500_data'
inference_base_dir = tmp / 'outputs/TF+500_data'
benchmark_output_dir = Path('outputs')
dirs = [refnets_base_dir, inference_base_dir, benchmark_output_dir]
[d.mkdir(exist_ok = True) for d in dirs]



all_stats = {}
# for dataset in [d.name for d in inference_base_dir.iterdir() if d.is_dir()]:
for dataset in ['replogle']:
	tmp = inference_base_dir / dataset
	for method in [d.name for d in tmp.iterdir() if d.is_dir()]:
	# for method in ['PPCOR']:

		anton_util.log_timestamp('prepping')

		try:
			estimated_network_edgelist = pd.read_csv(
				inference_base_dir / dataset / method / 'rankedEdges.csv',
				sep = '\t')
			reference_network_edgelist = pd.read_csv(
				refnets_base_dir / dataset / 'refNetwork.csv',
				sep = ',')
		except FileNotFoundError:
			s = f'Did not find required file for {dataset}/{method}, skipping..'
			print(s)
			continue


		estimated_network = functions.edgelist_to_matrix(np.array(
			estimated_network_edgelist
			))
		reference_network = functions.edgelist_to_matrix(np.array(
			reference_network_edgelist
			))

		# conditions = []
		# for axis in range(len(reference_network.shape)):
		# 	conditions.append(np.isin(
		# 		np.array(reference_network.axes[axis]),
		# 		np.array(estimated_network.axes[axis])
		# 		))
		# reference_network_filtered = (
		# 	reference_network.loc[tuple(conditions)]
		# 	)


		tmp = functions.harmonise_networks((
			estimated_network, 
			reference_network))
		harmonised_estimated_network, harmonised_reference_network = tmp



		# # Sanity check; Elements are still same after transformations
		# equals = []
		# # for ii in range(estimated_network_edgelist.shape[0]):  # Slow
		# import random
		# for ii in random.sample(
		# 		range(estimated_network_edgelist.shape[0]),
		# 		k = 1000):
		# 	r, t, v = estimated_network_edgelist.iloc[ii, :]
		# 	equals.append(
		# 		harmonised_estimated_network.loc[r, t] == v
		# 		)
		# assert(all(equals))

		# from IPython import embed; embed(colors = 'Linux')

		anton_util.log_timestamp('benching')

		run_id = f'{dataset}_{method}'

		# try:
		stats = gs.benchmarking.benchmark(
			estimated_network = harmonised_estimated_network,
			reference_network = harmonised_reference_network,
			plot_dir = (
				benchmark_output_dir / 'individual_beeline_methods'
				),
			method_name = run_id,
			fix_pr_ylim = False,
			)
		all_stats[run_id] = {
			'dataset': dataset,
			'method': method,
			**stats
			}
		# except Exception as e:
		# 	print(e)


		anton_util.log_timestamp(f'{dataset} x {method} finished')


df = pd.DataFrame(all_stats).T
df.index.name = 'inference_id'
df.to_csv(benchmark_output_dir / 'beeline_methods_stats.csv')


anton_util.log_timestamp('script finished')


