import json
items = None

def init_config(json_file):
    global items
    items = json.load(open(json_file, 'r'))

def get_source_max_length():
    algs = items['data']['max_src_length']
    if algs is None:
        raise IndexError("source_max_length must not be empty in configuration file")

    return algs


def get_accumulate_batches():
    algs = items['training'].get('accumulate_grad_batches',48) #64)
    if algs is None:
        raise IndexError("accumulate_grad_batches must not be empty in configuration file")

    return algs


def get_target_max_length():
    algs = items['data']['max_trg_length']
    if algs is None:
        raise IndexError("source_max_length must not be empty in configuration file")

    return algs


def get_batch_size():
    algs = items['data']['batch_size']
    if algs is None:
        raise IndexError("batch size must not be empty in configuration file")

    return algs


def get_learning_rate():
    algs = items['training']['lrate']
    if algs is None:
        raise IndexError("learning rate must not be empty in configuration file")

    return algs


def get_max_epochs():
    algs = items['training'].get('max_epochs',1000)
    if algs is None:
        raise IndexError("source_max_length must not be empty in configuration file")

    return algs


def get_model_name():
    algs = items['model']['type']
    if algs is None:
        raise IndexError("model name must not be empty in configuration file")

    return algs

