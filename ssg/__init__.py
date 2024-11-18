import argparse
import os
import reasoning_ros_interface.methods.ssg_3d.codeLib as codeLib
import torch
from pathlib import Path
from typing import Optional
from .training import Trainer
from . import dataset
from .sgfn import SGFN
from .sgpn import SGPN
from .jointSG import JointSG
from .imp import IMP

__all__ = ['SGFN', 'SGPN', 'dataset',
           'Trainer', 'IMP', 'JointSG']


def default_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str, default='./configs/config_default.yaml',
                        help='configuration file name. Relative path under given path (default: config.yml)')
    parser.add_argument('-m', '--mode', type=str, choices=['train', 'validation', 'trace', 'eval',
                        'sample', 'trace'], default='train', help='mode. can be [train,trace,eval]', required=False)
    parser.add_argument('--loadbest', type=int, default=0, choices=[
                        0, 1], help='1: load best model or 0: load checkpoints. Only works in non training mode.')
    parser.add_argument('--log', type=str, default='DEBUG',
                        choices=['DEBUG', 'INFO', 'WARNING', 'CRITICAL'], help='')
    parser.add_argument('-o', '--out_dir', type=str, default='',
                        help='overwrite output directory given in the config file.')
    parser.add_argument('--dry_run', action='store_true',
                        help='disable logging in wandb (if that is the logger).')
    parser.add_argument('--cache', action='store_true',
                        help='load data to RAM.')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--model_dir', type=str, default='')
    return parser


def load_config(args):
    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        raise RuntimeError(
            'Targer config file does not exist. {}'.format(config_path))

    # load config file
    config = codeLib.Config(config_path)
    # configure config based on the input arguments
    config.config_path = config_path
    config.LOADBEST = args.loadbest
    config.MODE = args.mode
    if len(args.out_dir) > 0:
        config.training.out_dir = args.out_dir
    if len(args.model_dir) > 0:
        config.training.model_dir = args.model_dir
    if args.dry_run:
        config.wandb.dry_run = True
    if args.cache:
        config.data.load_cache = True

    # check if name exist
    if 'name' not in config:
        config_name = os.path.basename(args.config)
        if len(config_name) > len('config_'):
            name = config_name[len('config_'):]
            name = os.path.splitext(name)[0]
            translation_table = dict.fromkeys(map(ord, '!@#$'), None)
            name = name.translate(translation_table)
            config['name'] = name

    # init device
    if torch.cuda.is_available() and len(config.GPU) > 0:
        config.DEVICE = torch.device("cuda")
    else:
        config.DEVICE = torch.device("cpu")

    config.log_level = args.log
    
    if len(args.data_path) > 0:
        config.data.path = args.data_path
        
    return config


def Parse():
    r"""loads model config

    """
    args = default_parser().parse_args()
    return load_config(args)



def get_config(config: Path,
               model_dir: Path,
               loadbest: bool = True,
               log: str = "NONE",
               out_dir: Optional[str] = None,
               mode: str = 'eval',
               dry_run: bool = True,
               load_cache: bool = False,
               data_path: str = '') -> codeLib.Config:
    """
    Initializes the config.
    :param config: Path to the config file.
    :param model_dir: Path to the model directory.
    :param loadbest: Load the best model or not.
    :param log: Log level.
    :param out_dir: Output directory.
    """
    config_path = os.path.abspath(str(config))
    if not os.path.exists(config_path):
        raise RuntimeError(
            'Target config file does not exist. {}'.format(config_path))

    # load config file
    config = codeLib.Config(config_path)
    # configure config based on the input arguments
    config.config_path = config_path
    config.LOADBEST = int(loadbest)
    config.MODE = mode
    config.training.model_dir = str(model_dir)
    config.wandb.dry_run = dry_run
    config.data.load_cache = load_cache
    
    if out_dir:
        config.training.out_dir = out_dir

    # check if name exist
    if 'name' not in config:
        config_name = os.path.basename(str(config))
        if len(config_name) > len('config_'):
            name = config_name[len('config_'):]
            name = os.path.splitext(name)[0]
            translation_table = dict.fromkeys(map(ord, '!@#$'), None)
            name = name.translate(translation_table)
            config['name'] = name

    # init device
    if torch.cuda.is_available() and len(config.GPU) > 0:
        config.DEVICE = torch.device("cuda")
    else:
        config.DEVICE = torch.device("cpu")

    config.log_level = log
    config.data.path = data_path
        
    return config
