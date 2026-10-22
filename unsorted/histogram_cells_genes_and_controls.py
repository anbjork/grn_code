


import anton_util
infile = 'preprocessed_data.pkl'
datasets = anton_util.unpickle_object(infile)
aa = []
aa = [ds['Y'].shape[0] for ds in datasets]
import matplotlib.pyplot as plt
plt.hist(aa)
plt.savefig('tmp_n_rows_in_data.png')
plt.close('all')
aa = [ds['Y'].shape[1] for ds in datasets]
plt.hist(aa)
plt.savefig('tmp_n_cols_in_data.png')
plt.close('all')
cc = []
for ds in datasets:
    y = ds['Y']
    nc = sum(y.index == 'control')
    cc.append(nc)
plt.hist(cc)
plt.savefig('tmp_n_controls_in_data.png')
plt.close('all')



