


def append_pickle(data, path):
    from pathlib import Path
    import anton_util
    p = Path(path)
    if p.exists():
        previous_data = anton_util.unpickle_object(path)
        data = previous_data + data
    else:
        p.parent.mkdir(exist_ok=True, parents=True)
    anton_util.pickle_object(data, path)
    anton_util.log_timestamp(f'total data length: {len(data)}')







def inference(
        data_path,
        output_path,
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

    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    anton_util.log_timestamp('Saving...')
    append_pickle(estimated_networks, output_path)










