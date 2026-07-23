


def append_pickle(data, path):
    from pathlib import Path
    import anton_util
    if Path(path).exists():
        previous_data = anton_util.unpickle_object(path)
        data = previous_data + data
    anton_util.log_timestamp(f'total data: {len(data)}')
    anton_util.pickle_object(data, path)







def inference(
        data_path,
        output_dir_path,
        method_function):

    from pathlib import Path
    import anton_util
    import copy

    path = data_path
    data_sources = anton_util.unpickle_object(path)

    estimated_networks = []
    for ii, data_source in enumerate(data_sources):
        anton_util.log_timestamp(f'dataset {ii}...')
        anton_util.log_timestamp(repr(method_function))

        ens = method_function(data=data_source)
        for method_name, estimated_network in ens.items():
            meta = copy.deepcopy(data_source['meta'])
            meta['method'] = method_name
            estimated_networks.append({
                'meta': meta,
                'estimated_network': estimated_network,
                })

    output_dir = Path(output_dir_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    p = output_dir / f'estimated_networks.pkl'

    anton_util.log_timestamp('Saving...')
    append_pickle(estimated_networks, p)










