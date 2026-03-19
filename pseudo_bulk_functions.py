import numpy as np 






def pseudo_bulk_group(Y):
    """
    Pseudo bulks a chunk of samples belonging to the same condition
    """

    n_pseudo_bulks = 3

    from math import floor
    chunk_size = floor(Y.shape[0] / n_pseudo_bulks)
    chunk_indices = chunk_size * np.array(range(n_pseudo_bulks))
    chunks = []
    for ii in chunk_indices:
        chunk = Y.iloc[ii : ii + chunk_size, :]
        chunks.append(chunk)
    # Redo the last chunk to include the remainder.
    # A chunk that is slightly bigger at the end should be much
    # better than a small remainder chunk, for the statistical properties
    # of the pseudo bulks. Size difference should be negligible too
    chunks.pop()
    chunks.append(Y.iloc[chunk_indices[-1] : , :])
    #
    # Probably a smarter way, except that the last chunk is not handled.
    # Maybe inspiration for improvement
    # //AB
    # chunks = [
    #     Y[i : i + chunk_size, :]
    #     for i in range(0, Y.shape[0], chunk_size)
    #     ]

    tmp = [f'psb{i}' for i in range(n_pseudo_bulks)]
    pseudo_bulk = {l: chunk.sum(axis = 0) for l, chunk in zip(tmp, chunks)}
    return pseudo_bulk



def pseudo_bulk(Y, P):
    """
    Y is a pandas data frame of expression values
    P is a pandas data frame of perturbation values
    """

    # Based on initial testing, pseudo bulking (like this) is a 
    # significant part of the runtime for the lsco method
    pseudo_bulks = {}
    for perturbed_gene in P:
        perturbed_cell_indices = np.nonzero(P[perturbed_gene])[0]
        perturbed_cells = Y.iloc[perturbed_cell_indices, :]
        group = pseudo_bulk_group(perturbed_cells)
        for psl, psb in group.items():
            pseudo_bulks[f'{perturbed_gene}_{psl}'] = psb

    import pandas as pd
    df = pd.DataFrame.from_dict(pseudo_bulks, orient = 'index')

    return df


