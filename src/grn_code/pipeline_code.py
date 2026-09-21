

def get_pipeline_path(output_base_path):
    pipeline_base_path = output_base_path / 'in_pipeline'
    return pipeline_base_path



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

    f_name = method_function.__name__

    path = data_path
    data_sources = anton_util.unpickle_object(path)
    # data_sources = [data_sources[3]] # Debug

    estimated_networks = []
    for ii, data_source in enumerate(data_sources):
        anton_util.log_timestamp(f'dataset {ii}...')
        anton_util.log_timestamp(f_name)

        try:
            ens = method_function(data=data_source)
            error = None
        except Exception as e:
            error = repr(e)
            anton_util.log_timestamp(
                    f'Error in method {f_name}: {error}')
            ens = {f_name: None}
        # Debug
        # ens = method_function(data=data_source)
        # error = None

        for method_name, estimated_network in ens.items():
            meta = copy.deepcopy(data_source['meta'])
            meta['method'] = method_name
            meta['inference error'] = error
            estimated_networks.append({
                'meta': meta,
                'estimated_network': estimated_network,
                })

    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    anton_util.log_timestamp('Saving...')
    append_pickle(estimated_networks, output_path)










