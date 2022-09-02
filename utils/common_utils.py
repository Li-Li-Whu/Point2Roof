import yaml
import torch
import logging
from pathlib import Path
from easydict import EasyDict
import torch.distributed as dist
import torch.multiprocessing as mp


def cfg_from_yaml_file(cfg_file):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)

    cfg = EasyDict(new_config)
    cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    return cfg


def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            logger.info('\n%s.%s = edict()' % (pre, key))
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, val))


def init_dist_pytorch(batch_size, local_rank, backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)
    dist.init_process_group(backend=backend)
    assert batch_size % num_gpus == 0, 'Batch size should be matched with GPUS: (%d, %d)' % (batch_size, num_gpus)
    batch_size_each_gpu = batch_size // num_gpus
    rank = dist.get_rank()
    return batch_size_each_gpu, rank


def get_dist_info():
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def create_logger(log_file=None, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger