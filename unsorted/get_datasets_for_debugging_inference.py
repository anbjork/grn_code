



import anton_util
import pandas as pd
from copy import deepcopy

infile = 'inferences.pkl'
inferences_unmodified = anton_util.unpickle_object(infile)

inferences = deepcopy(inferences_unmodified)
metas = [d['meta'] for d in inferences]

for meta in metas:
    for k, v in meta['dataset_parameters'].items():
        meta[k] = v
    meta.pop('dataset_parameters')

df = pd.DataFrame(metas)
dp = df.loc[df['method'] == 'psgrn', :]
de = df.loc[df['method'] == 'psgrn_inference', :]

die = de.loc[['index' in elem for elem in de['inference error']], :]
eii = set(die.index)
emetas = []
for ii, elem in enumerate(metas):
    if ii in eii:
        emetas.append(elem)

dataset_keys = []
inference_keys = ['inference error', 'method']
for emeta in emetas:
    for k in inference_keys:
        emeta.pop(k)
    dataset_keys.append(frozenset(emeta.items()))

datasets = anton_util.unpickle_object('preprocessed_data.pkl')

datameta = deepcopy([ds['meta'] for ds in datasets])
for meta in datameta:
    for k, v in meta['dataset_parameters'].items():
        meta[k] = v
    meta.pop('dataset_parameters')

datametakeys = [frozenset(dm.items()) for dm in datameta]
dataset_dict = {k: ds for k, ds in zip(datametakeys, datasets)}

edatasets = [dataset_dict[dk] for dk in dataset_keys]

anton_util.pickle_object(edatasets, 'psgrn_error_datasets.pkl')




edatasets = anton_util.unpickle_object('psgrn_error_datasets.pkl')


from grn_code.inference_functions import psgrn_inference

for data in edatasets:
    psgrn_inference(data=data)












