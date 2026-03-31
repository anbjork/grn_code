import anton_util
import pandas as pd


df = anton_util.unpickle_object('benchmarks/perturbation_methods_stats.pkl')


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

df = df.drop('f1_scores', axis = 1)

dfd = df.reset_index(drop = True)



