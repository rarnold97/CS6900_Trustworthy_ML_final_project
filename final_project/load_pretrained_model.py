from pathlib import Path
from typing import Tuple
import argparse
import re
import datetime
from dataclasses import dataclass
import os
from easydict import EasyDict
from logging import Logger

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from pcdet.datasets import build_dataloader, DistributedSampler, DatasetTemplate
from pcdet.config import cfg, cfg_from_yaml_file, cfg_from_list
from pcdet.utils import common_utils
from pcdet.models import build_network
from pcdet.models.detectors.pointpillar import PointPillar

if not torch.cuda.is_available():
    raise RuntimeError('Cannot operate code without CUDA ...')

PROJECT_ROOT: Path = Path(__file__).parent.parent.absolute()
BATCH_SIZE: int = int(os.environ.get('BATCH_SIZE', 1))
CONFIG_FILE: Path = Path(os.environ.get('CONFIG_FILE',\
    PROJECT_ROOT/"OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml"
))
CHECKPOINT_FILE: Path = Path(os.environ.get('CKPT',\
    PROJECT_ROOT/"checkpoints/pointpillar_7728.pth"))

TOOLS_PATH: Path = PROJECT_ROOT / "OpenPCDet" / "tools"
assert TOOLS_PATH.is_dir()

# the code expects the code path to be the TOOLS directory
os.chdir(TOOLS_PATH)

assert BATCH_SIZE, \
    'Please define an environment variable for BATCH_SIZE. ex: export BATCH_SIZE=4'

assert CONFIG_FILE, \
    'Please define a yaml config file for the pointpillar model.'
    
assert CHECKPOINT_FILE, \
    'Please define a checkpoint file for the pretrained moddel.'

USE_DISTRIBUTED_TESTING: bool = False


class OpenPcdetWrapper:
    """
    This class acts as a namespace that takes code from openpcdet that
    is not part of a module or package, without needing to completely refactor it.
    """
    @staticmethod
    def parse_config()->Tuple[argparse.Namespace, EasyDict]:
        parser = argparse.ArgumentParser(description='arg parser')
        parser.add_argument('--cfg_file', type=str, default=CONFIG_FILE, help='specify the config for training')

        parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, required=False, help='batch size for training')
        parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
        parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
        parser.add_argument('--ckpt', type=Path, default=CHECKPOINT_FILE, help='checkpoint to start from')
        parser.add_argument('--pretrained_model', type=str, default='PointPillar', help='pretrained_model')
        parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
        parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
        parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
        parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                            help='set extra config keys if needed')

        parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
        parser.add_argument('--start_epoch', type=int, default=0, help='')
        parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
        parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
        parser.add_argument('--ckpt_dir', type=Path, default=CHECKPOINT_FILE.parent.absolute(), help='specify a ckpt directory to be evaluated if needed')
        parser.add_argument('--save_to_file', action='store_true', default=False, help='')
        parser.add_argument('--infer_time', action='store_true', default=False, help='calculate inference latency')

        args = parser.parse_args()

        cfg_from_yaml_file(args.cfg_file, cfg)
        cfg.TAG = args.cfg_file.stem
        cfg.EXP_GROUP_PATH = args.cfg_file.relative_to(TOOLS_PATH / "cfgs").parent
        #cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

        np.random.seed(1024)

        if args.set_cfgs is not None:
            cfg_from_list(args.set_cfgs, cfg)

        return args, cfg
    
    @staticmethod
    def statistics_info(cfg, ret_dict, metric, disp_dict):
        for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
            metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
            metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
        metric['gt_num'] += ret_dict.get('gt', 0)
        min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
        disp_dict['recall_%s' % str(min_thresh)] = \
            '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])

@dataclass
class PreTrainedParams:
    model: nn.Module = None
    dataset_template: DatasetTemplate = None
    class_names: Tuple[str] = None
    cfg: EasyDict = None
    logger: Logger = None
            
def load_pretained_params()->PreTrainedParams:
  
    args, cfg = OpenPcdetWrapper.parse_config()

    args.batch_size = BATCH_SIZE

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    if not args.eval_all:
        num_list = re.findall(r'\d+', str(args.ckpt)) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    test_dataset, _, _ = \
    build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=USE_DISTRIBUTED_TESTING, workers=args.workers, logger=logger, training=False
    )

    model: PointPillar = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), \
        dataset=test_dataset)
    
    model.load_params_from_file(filename=CHECKPOINT_FILE, logger=logger, to_cpu=USE_DISTRIBUTED_TESTING)
    model.cuda()
    
    return PreTrainedParams(model, test_dataset, tuple(cfg.CLASS_NAMES), cfg, logger)

def test_pretrained_model_load():
    params = load_pretained_params
    
    assert params.model and params.data_loader and params.sampler \
        and params.dataset_template and params.class_names, \
        'Invalid pre-trained parameter set, error occurred loading model and parameters.'
