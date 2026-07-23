


def append_pickle(data, path):
    from pathlib import Path
    import anton_util
    if Path(path).exists():
        previous_data = anton_util.unpickle_object(path)
        data = previous_data + data
    anton_util.log_timestamp(f'total data: {len(data)}')
    anton_util.pickle_object(data, path)




