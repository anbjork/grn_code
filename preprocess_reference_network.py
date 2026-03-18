

import genesnake
import pandas as pd
import numpy as np

filename = 'data/beeline/networks/Networks/human/Non-specific-ChIP-seq-network.csv'
net = pd.read_csv(filename, sep=',')

net_mat = genesnake.util.edgelist_to_matrix(
    edgelist = np.array(net),
    )

import anton_util
anton_util.pickle_object(net_mat, 'data/Non-specific-ChIP-seq-network_with_weights.pkl')

net_mat.sort_index(inplace=True)
net_mat.sort_index(inplace=True, axis=1)







other_net = anton_util.unpickle_object('/home/anbjork/projects/small/replogle_grns/versions/33/replogle_round_2/data/Non-specific-ChIP-seq-network_with_weights.pkl')

other_net.sort_index(inplace=True)
other_net.sort_index(inplace=True, axis=1)


print('All elements in new net equals old net after sorting: ')
print((net_mat == other_net).all().all())




