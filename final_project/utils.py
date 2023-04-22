# source class to move away from in untargeted attack
from __future__ import annotations

from collections import namedtuple
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from functools import wraps
import pickle

import torch
from torch import Tensor

from final_project.load_pretrained_model import PROJECT_ROOT

__all__ = [
    "SOURCE_CLASS",
    "CLASS_MAP",
    "CACHE_DIR",
    "SAMPLE_SIZE",
    "SOURCE_LABEL",
    "DENSE_TENSOR_INPUT",
    "cache_decorator",
    "KittiLabel",
    "InputTensor",
    "CacheType"
]

SOURCE_LABEL: str = 'Pedestrian'
ClassMap = namedtuple('ClassMap', ['car', 'pedestrian', 'cyclist'])
CLASS_MAP = ClassMap(0, 1, 2)

CACHE_DIR: Path = PROJECT_ROOT.parent.absolute()/"cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DENSE_TENSOR_INPUT: str = 'spatial_features_2d'
BATCH_SIZE = 1
SAMPLE_SIZE: int = 50


class CacheType(Enum):
    PKL = 0
    TORCH = 1

def cache_decorator(cache_path: Path, cache_type: CacheType):
    """
    This is a decorator that will cache any object returned by
    the function it encapsulates.  This is helpful, since
    the dataset is large (>50GB), and subsequently the 
    processed results are also large.

    Parameters
    ----------
    cache_path : Path
        where the cache file should be.
    cache_type : CacheType
        whether to cache using pickle or torch.

    Returns
    -------
    Any
        Returns the object that was cached

    Raises
    ------
    ValueError
        throws exception if cache file does not exist.
    """

    def inner_function(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            if cache_path.exists():
                with open(cache_path, 'rb') as file:
                    print(f'___ LOADING CACHE FOR: {cache_path} ___')
                    return pickle.load(file) if cache_type == CacheType.PKL else torch.load(file)
            else:
                object = function(*args, **kwargs)
                with open(cache_path, 'wb') as file:
                    if cache_type == CacheType.PKL:
                        pickle.dump(object, file)
                    elif cache_type == CacheType.TORCH:
                        torch.save(object, file)
                    else:
                        raise ValueError('Unsupported caching type ...')
                return object
        return wrapper
    return inner_function


def get_encoded_cls_label(annotations: dict)->Tensor:
    """
    converts labeled data to be numerical

    Parameters
    ----------
    annotations : dict
        _description_

    Returns
    -------
    Tensor
        tensor of numerical labels   
    """
    assert 'name' in annotations
    raw_labels = annotations['name']
    encoded_labels = [getattr(CLASS_MAP, label.lower(), 3) for label in raw_labels]
    assert len(encoded_labels) == len(annotations['name'])
    return torch.tensor(encoded_labels).to(torch.int32)
    

@dataclass
class OriginalData:
    sample_idx: int = 0
    batch_index: int = 0
    batch_data: dict = None
    orig_annotations: dict = None
    feature_data: Tensor = None
    ground_truth: dict = None
    frame_id: str = ''

@dataclass
class InputTensor:
    gt_boxes: Tensor = None
    batch_size: int = 1
    X: Tensor = None
    frame_id: str = ''
    
    @classmethod
    def from_batch_dict(cls, data: dict, frame_id: str):
        return cls(
            data['gt_boxes'],
            1,
            data['spatial_features_2d'],
            frame_id
        )

    def __call__(self)->dict:
        return {
            'gt_boxes': self.gt_boxes,
            'batch_size': self.batch_size,
            'spatial_features_2d': self.X
        }