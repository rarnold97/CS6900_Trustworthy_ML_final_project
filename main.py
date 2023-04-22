from __future__ import annotations

import typing
from pathlib import Path
from functools import partial
from dataclasses import dataclass, field
import time
from collections import namedtuple
import tabulate

import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import plotly.graph_objects as go

import tools.visual_utils.open3d_vis_utils as V
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.datasets.kitti.kitti_object_eval_python.eval import eval_class
from pcdet.models.backbones_2d.base_bev_backbone import BaseBEVBackbone
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from final_project.load_pretrained_model import load_pretained_params, PreTrainedParams, BATCH_SIZE
from final_project.attack import ParamsPGD, AttackParams, ParamsDeepFool, AdvAttack
from final_project.utils import get_encoded_cls_label, cache_decorator, CacheType, OriginalData
from final_project.utils import CACHE_DIR, SAMPLE_SIZE, BATCH_SIZE, SOURCE_LABEL, SAMPLE_SIZE, CLASS_MAP, PROJECT_ROOT


@cache_decorator(CACHE_DIR / "pretrained_params.pkl", CacheType(CacheType.PKL))
def load_pretrained():
    return load_pretained_params()

PRETRAIN = load_pretrained()
PRETRAIN.model.eval()


@cache_decorator(CACHE_DIR / "data_info.pkl", CacheType.PKL)
def load_kitti_dataset():
    kitti_dataset = KittiDataset(
        PRETRAIN.cfg.DATA_CONFIG,
        PRETRAIN.class_names,
        training=False,
        root_path=Path(PRETRAIN.cfg.DATA_CONFIG.DATA_PATH),
        logger=PRETRAIN.logger
    )
    
    return kitti_dataset.get_infos()
    
GT_DATA = load_kitti_dataset()
GT_MAP = {}

for idx, truth_data in enumerate(GT_DATA):
    GT_MAP[truth_data['image']['image_idx']] = idx


class AdvSamplesDataset(Dataset):
    """
    Used as an override for the pytorch dataset/dataloader paradigm

    Parameters
    ----------
    Dataset : torch.utils.Dataset
        dataset in pytorch format
    """
    def __init__(self, source_labeled_data: typing.List[typing.Tuple[dict, dict]]):
        self.labeled_pairs = [(x, y) for x,y in source_labeled_data]
    
    def __len__(self):
        return len(self.labeled_pairs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            return (self.labeled_pairs[i.item()] for i in idx)
        else:
            return self.labeled_pairs[idx]

def dict_to_cpu(d: dict)->None:
    for key, value in d.items():
        if torch.is_tensor(value):
            d[key] = value.cpu()


@dataclass
class AdvResults:
    """
    Used to store the adversarial examples and
    the data used to generate it.
    """
    name: str = ''
    adv_examples: typing.List[Tensor] = field(default_factory=lambda: [])
    result_summaries: typing.List[dict] = field(default_factory=lambda: [])
    orig_data: OriginalData = field(default_factory=lambda: [])


@cache_decorator(CACHE_DIR / "pretrained_params.pkl", CacheType(CacheType.PKL))
def load_pretrained():
    return load_pretained_params()
    

def generate_adv_examples()->None:
    """
    acts as the main entrypoint for the project that
    pulls all the code together.
    """
    checkpoint_state = PRETRAIN.model.dense_head.state_dict()
    
    def forward_spatial2d(pointpillar: nn.Module, batch_dict: dict)->typing.Tuple[dict, dict]:
        """
        uses the prediction pipeline that is implemented in pcdet.
        Accepts the batch parameters dict that is schemed in
        the dataset/dataloader, and returns annotated output.

        Parameters
        ----------
        batch_dict : dict
            model input parameters for point pillar.

        Returns
        -------
        Tuple[dict, dict[]
            contains performance metrics, as well as predictions.
            Also returns a second dictionary with statistic calculations
        """
        load_data_to_gpu(batch_dict)

        with torch.no_grad():
            for mod in pointpillar.module_list:
                batch_dict = mod(batch_dict)
                if isinstance(mod, BaseBEVBackbone):
                    break
        
        return batch_dict

    @cache_decorator(CACHE_DIR / "adv_input_samples.bin", CacheType.TORCH)
    def load_adv_input_samples()-> AdvSamplesDataset:
        """
        used to load data to be perturbed.

        Returns
        -------
        AdvSamplesDataset
            returns a class that contains the dataset
            to be perturbed
        """

        # randomly sample input data, and collect samples that actually predict pedestrian
        data_loader_shuffled = DataLoader(
            PRETRAIN.dataset_template,
            batch_size=BATCH_SIZE,
            pin_memory=True,
            num_workers=4,
            shuffle=False,
            collate_fn=PRETRAIN.dataset_template.collate_batch,
            drop_last=False,
            sampler=None,
            timeout=0,
            worker_init_fn=partial(common_utils.worker_init_fn, seed=None)
        )

        # extract the spatial features, and predict them as classes
        source_data_samples = []
        
        for sample_index, batch_dict in enumerate(data_loader_shuffled):
            spatial_data_dict = forward_spatial2d(PRETRAIN.model, batch_dict)
            # forward propagate the dense head to get 2d class name predictions
            # pass in refined feature dictionary to prevent extra copying
            # and conserve GPU memory
            load_data_to_gpu(spatial_data_dict)
            with torch.no_grad():
                dense_output = PRETRAIN.model.dense_head.forward(
                    {'gt_boxes': spatial_data_dict['gt_boxes'], 
                    'spatial_features_2d': spatial_data_dict['spatial_features_2d'], 
                    'batch_size': spatial_data_dict['batch_size']}
                )

            pred_dicts, _ = PRETRAIN.model.post_processing(dense_output)
            annos = PRETRAIN.dataset_template.generate_prediction_dicts(
                batch_dict,
                pred_dicts,
                PRETRAIN.class_names,
                None
            )
            
            del dense_output

            gt_data = GT_DATA[sample_index]
            
            for batch_idx, anno in enumerate(annos):
                if SOURCE_LABEL in gt_data['annos']['name']:
                    # exract at batch index we are interested in.
                    
                    dict_to_cpu(spatial_data_dict)
                    dict_to_cpu(batch_dict)
                    
                    data_sample =  OriginalData(
                        sample_idx=sample_index,
                        batch_index=batch_idx,
                        frame_id=batch_dict['frame_id'][batch_idx],
                        orig_annotations=anno,
                        batch_data=batch_dict,
                        feature_data=spatial_data_dict['spatial_features_2d'].cpu(),
                        ground_truth=gt_data
                    )
                    source_data_samples.append((data_sample, anno))

            if len(source_data_samples) >= SAMPLE_SIZE:
                break

        del data_loader_shuffled
        del spatial_data_dict

        return AdvSamplesDataset(source_data_samples)

    adv_input_samples = load_adv_input_samples()

    #TODO add collate function to extract dense head only data
    def collate(samples):
        """
        transform the data dictionaries provided by
        pcdet into proper tensors for attack algorithms

        Parameters
        ----------
        samples : AdvSamplesDataset
            contains filtered adversarial inputs

        Returns
        -------
        Tuple[Tensor, Tensor]
            returns feature data with corresponding classification labels.
        """
        X = [data for data, _ in samples]
        labels = [get_encoded_cls_label(annotation) for _, annotation in samples]
        return X, labels
    
    refined_dataloader = DataLoader(
        adv_input_samples, 
        batch_size=BATCH_SIZE,
        pin_memory=False,
        sampler=None,
        drop_last=False,
        timeout=0,
        shuffle=False,
        collate_fn=collate
        )    
    
    def adv_ex_decorator(func):
        def wrapper():
            """
            wrapper for attack methods so they can be cached later

            Parameters
            ----------
            func : Callable
                Has the following signature:
                x: Tensor, label: Tensor
            """
            start = time.time()
            # execute the adversarial attacks
            adv_examples = AdvResults()
            
            for batch_sample_idx, (X, _) in enumerate(refined_dataloader):
                # iterate throught the batches
                print(f'PROCESSING BATCH NUMBER: {batch_sample_idx+1}')
                for x in X:
                    # function is expected to write to results container
                    x_adv, summary = func(x)
                    PRETRAIN.model.dense_head.load_state_dict(checkpoint_state)
                    
                    adv_examples.adv_examples.append(x_adv)
                    adv_examples.result_summaries.append(summary)
                    adv_examples.orig_data.append(x) 

            end = time.time()
            print(f'CURRENT ATTACK TOOK: {end-start} [s]')
            return adv_examples
        return wrapper
    

    @cache_decorator(CACHE_DIR / f"{SAMPLE_SIZE}_pgd_adv_examples.bin", CacheType.TORCH)
    def generate_pgd_examples():
        
        @adv_ex_decorator
        def execute_pgd_attack(data: OriginalData)->typing.Tuple[Tensor, dict]:
            params = ParamsPGD(
                class_names=PRETRAIN.class_names,
                step_size=0.001,
                max_iters=10,
                epsilon=0.5
            )
            return AdvAttack.pgd_attack_pcdet(data, PRETRAIN.model, params)
        
        pgd_adv_examples = execute_pgd_attack()
        pgd_adv_examples.name = 'pgd'
        return pgd_adv_examples
    
    deepfool_examples_cache_path = CACHE_DIR / f"{SAMPLE_SIZE}_deepfool_adv_examples.bin"
    @cache_decorator(deepfool_examples_cache_path, CacheType.TORCH)
    def generate_deepfool_examples():
        
        @adv_ex_decorator
        def execute_deepfool_attack(data: OriginalData)->typing.Tuple[Tensor, dict]:
            params = ParamsDeepFool(
                class_names=PRETRAIN.class_names,
                max_iters=10
            )
            params.overshoot = 0.01
            
            return AdvAttack.deepfool_pcdet(data, PRETRAIN.model, params)  
        
        deepfool_adv_examples = execute_deepfool_attack()
        deepfool_adv_examples.name = 'deepfool'
        return deepfool_adv_examples
    
    fgsm_cache_path = CACHE_DIR / f'{SAMPLE_SIZE}_fgsm_adv_examples.bin'
    @cache_decorator(fgsm_cache_path, CacheType.TORCH)
    def generate_fgsm_examples():
        
        @adv_ex_decorator
        def execute_fgsm_attack(data: OriginalData)->typing.Tuple[Tensor, dict]:
            params = AttackParams(
                class_names=PRETRAIN.class_names,
                max_iters=1,
                epsilon=0.01
            )
            return AdvAttack.fgsm_attack_pcdet(data, PRETRAIN.model, params)
    
        fgsm_adv_examples = execute_fgsm_attack()
        fgsm_adv_examples.name = 'fgsm'
        return fgsm_adv_examples
    
    print('!!!!!!!!!! PERFORMING PGD ATTACK !!!!!!!!!!')
    pgd_adv_examples = generate_pgd_examples()
    print(f'!!!!!!!!!! END PGD ATTACK !!!!!!!!!!')
    del pgd_adv_examples
    
    print("^^^^^ INITIATING DEEP FOOL ATTACK ^^^^^")        
    deepfool_adv_examples = generate_deepfool_examples()
    print("^^^^^ END DEEP FOOL ATTACK ^^^^^")
    del deepfool_adv_examples
    
    print("----- INITIATING FGSM ATTACK -----")        
    fgsm_adv_examples = generate_fgsm_examples()
    print("----- END FGSM ATTACK -----")  
    del fgsm_adv_examples


@dataclass
class Stats:
    """
    Used to collect statistics about perturbed feature data.
    """
    recall: np.ndarray = np.array([])
    accuracy: np.ndarray = np.array([])
    recall_avg: float = 99999
    accuracy_avg: float = 99999
    
@dataclass
class ProbHisto:
    """
    Stores datasets that will be plotted as histogram distributions.
    """
    probs_car: typing.List[float] = field(default_factory=lambda: [])
    probs_pedestrian: typing.List[float] = field(default_factory=lambda: [])
    probs_cyclist: typing.List[float] = field(default_factory=lambda: [])
    
    avg_car: float = 0.0
    avg_pedestrain: float = 0.0
    avg_cyclist: float = 0.0
    
    def calc_avg(self):
        self.avg_car = np.array(self.probs_car).mean()
        self.avg_cyclist = np.array(self.probs_cyclist).mean()
        self.avg_pedestrain = np.array(self.probs_pedestrian).mean()
        
def profile(dataset_cache_path: Path)->\
    typing.Tuple(Stats, Stats, Stats, ProbHisto, ProbHisto):
    """
    Calculates the metrics of the adversarial data
    to assess attack performance.

    Parameters
    ----------
    dataset_cache_path : Path
        the path to the cached dataset. Caching is
        neccessary because each dataset is >10GB.

    Returns
    -------
    tuple
        returns statistics and data for histograms.
        
        three stats objects are returned:
        1-stats for original prediction
        2-stats for adversarial prediction
        3-stats for using the original prediction as ground truth
        
        two histogram datasets are returned:
        1-histogram data for original dataset predictions.
        2-histogram data for adversarial dataset predictions.
    """

    assert dataset_cache_path.exists()

    adv_data: AdvResults = None
    with open(dataset_cache_path, 'rb') as file:
        adv_data = torch.load(file)
 
    stats_orig = Stats()
    stats_adv = Stats()
    stats_pred = Stats()
    prob_histo_orig = ProbHisto()
    prob_histo_adv = ProbHisto()
    
    accuracy_orig, recall_orig = [], []
    accuracy_adv, recall_adv = [], []
    accuracy_pred, recall_pred = [], []
    
    num_missmatch_gt: int = 0
    num_missmatch_orig_pred: int = 0

    def update_histo(pred_scores: Tensor, pred_labels: Tensor, histo: ProbHisto):
        for  p, label in zip(pred_scores, pred_labels):
            if getattr(CLASS_MAP, 'car') == label:
                histo.probs_car.append(p.item())
            elif getattr(CLASS_MAP, 'pedestrian') == label:
                histo.probs_pedestrian.append(p.item())
            elif getattr(CLASS_MAP, 'cyclist') == label:
                histo.probs_cyclist.append(p.item())

    for i, (predictions, orig_data) in enumerate(zip(adv_data.result_summaries, adv_data.orig_data)):

        update_histo(predictions['pred_scores'], predictions['pred_labels']-1, prob_histo_adv)
        update_histo(orig_data.orig_annotations['score'], get_encoded_cls_label(orig_data.orig_annotations), \
            prob_histo_orig)
        
        if len(orig_data.orig_annotations['name']) != len(predictions['pred_labels']) or \
            len(predictions['pred_labels']) != len(orig_data.ground_truth['annos']['name']):
            num_missmatch_gt += 1
        else:
            orig_encoded = get_encoded_cls_label(orig_data.orig_annotations)
            encoded_label = getattr(CLASS_MAP, SOURCE_LABEL.lower(), 3)
            gt_encoded =get_encoded_cls_label(orig_data.ground_truth['annos'])
            source_mask = gt_encoded == encoded_label
            
            if not source_mask.any():
                print(f'EVAL #{i+1}: SKIPPING example without predicted: {SOURCE_LABEL}')
                continue
            
            N = float(len(gt_encoded))
            tp_plus_fn = source_mask.sum()
            
            predict_encoded = predictions['pred_labels'] - 1
            predict_source_samples = predict_encoded[source_mask]
            orig_samples = orig_encoded[source_mask]
            
            predict_tp = (predict_source_samples.cpu() == gt_encoded[source_mask].cpu()).sum()
            orig_tp = (orig_samples == gt_encoded[source_mask]).cpu().sum()
            
            rec_adv = predict_tp / tp_plus_fn
            rec_orig = orig_tp / tp_plus_fn
            
            acc_adv = (predict_encoded.cpu() == gt_encoded.cpu()).sum().numpy() / N
            acc_orig = (orig_encoded.cpu() == gt_encoded.cpu()).sum().numpy() / N
            
            accuracy_orig.append(rec_orig.item())
            recall_orig.append(acc_orig.item())
            
            accuracy_adv.append(acc_adv)
            recall_adv.append(rec_adv)
        
        if len(orig_data.orig_annotations['name']) == len(predictions['pred_labels']):
            orig_encoded = get_encoded_cls_label(orig_data.orig_annotations)
            encoded_label = getattr(CLASS_MAP, SOURCE_LABEL.lower(), 3)
            source_mask = orig_encoded == encoded_label
            predict_encoded = predictions['pred_labels'] - 1
            
            N = float(len(source_mask))
            tp = (predict_encoded[source_mask].cpu() == orig_encoded[source_mask].cpu())\
                .sum().numpy()
                
            recall_pred_truth = tp / N
            acc_pred_truth = (predict_encoded.cpu() == orig_encoded.cpu())\
                .sum().numpy() / N
                
            accuracy_pred.append(acc_pred_truth)
            recall_pred.append(recall_pred_truth)

            num_missmatch_orig_pred += 1
            
    stats_orig.accuracy = np.array(accuracy_orig)
    stats_orig.recall = np.array(recall_orig)
    stats_adv.accuracy = np.array(accuracy_adv)
    stats_adv.recall = np.array(recall_adv)
    stats_pred.accuracy = np.array(accuracy_pred)
    stats_pred.recall = np.array(recall_pred)
    
    stats_orig.accuracy_avg = stats_orig.accuracy.mean()
    stats_orig.recall_avg = stats_orig.recall.mean()
    stats_adv.accuracy_avg = stats_adv.accuracy.mean()
    stats_adv.recall_avg = stats_adv.recall.mean()
    stats_pred.accuracy_avg = stats_pred.accuracy.mean()
    stats_pred.recall_avg = stats_pred.recall.mean()
    
    prob_histo_adv.calc_avg()
    prob_histo_orig.calc_avg()
    
    print(f'INSTANCES OF MISSMATCH WITH GT: {num_missmatch_gt}')
    print(f'INSTANCES OF MISSMATCH WITH ORIG PREDICT: {num_missmatch_orig_pred}')
    return stats_orig, stats_adv, stats_pred, prob_histo_orig, prob_histo_adv

def plot_example(cache_filename: Path):
    """
    Plots input LiDAR data scenes with a 3D renderer.
    Currently does not work though. Functions from provided
    API are likely not working in because I developed
    in a WSL environment.

    Parameters
    ----------
    cache_filename : Path
        filename of cached data.
    """
    adv_data: AdvResults = None
    with open(cache_filename, 'rb') as file:
        adv_data = torch.load(file)
         
    kitti_dataset = KittiDataset(
        PRETRAIN.cfg.DATA_CONFIG,
        PRETRAIN.class_names,
        training=False,
        root_path=Path(PRETRAIN.cfg.DATA_CONFIG.DATA_PATH),
        logger=PRETRAIN.logger
    )
    found_good_ex: bool = False
    
    while not found_good_ex:
            rand_idx = np.random.randint(low=0, high=len(adv_data.adv_examples)-1)
            
            orig_data = adv_data.orig_data[rand_idx]
            PRETRAIN.model.forward(orig_data.batch_data)
            prediction = adv_data.result_summaries[rand_idx]
            sample_idx = GT_MAP[orig_data.frame_id]
            gt_info = GT_DATA[sample_idx]
            
            if SOURCE_LABEL not in gt_info['annos']['name']:
                continue
            else:
                break
        
    data_dict = kitti_dataset[sample_idx]

    # DOES NOT WORK WITH WSL
    # TODO Test on platform other than WSL
    V.draw_scenes(
        data_dict['points'][:, 1:],
        data_dict['gt_boxes'], 
        ref_boxes = prediction['pred_boxes'],
        ref_labels=prediction['pred_labels'],
        ref_scores=prediction['pred_scores']
    )


def main()->None:
    CacheFiles = namedtuple('CacheFiles', ['fgsm', 'pgd', 'deepfool'])
    expected_cache_files = CacheFiles(
        CACHE_DIR / "50_fgsm_adv_examples.bin",
        CACHE_DIR / "50_pgd_adv_examples.bin",
        CACHE_DIR / "50_deepfool_adv_examples.bin"    
    )
    
    for cache_file in expected_cache_files:
        if not cache_file.exists():
            generate_adv_examples()

    stats_fgsm_orig, stats_fgsm_adv, stats_fgsm_pred, histo_orig_fgsm, histo_adv_fgsm = \
        profile(expected_cache_files.fgsm)
    stats_deepfool_orig, stats_deepfool_adv, stats_deepfool_pred, _, histo_adv_deepfool = \
        profile(expected_cache_files.deepfool)
    stats_pgd_orig, stats_pgd_adv, stats_pgd_pred, _, histo_adv_pgd = \
        profile(expected_cache_files.pgd)
    
    def pct_diff(value, orig):
        return 100 * (value - orig) / orig
   
    if len(stats_fgsm_adv.accuracy) == 0:
        acc_pct_diff_fgsm = pct_diff(stats_fgsm_adv.accuracy_avg, stats_fgsm_orig.accuracy_avg)
        recall_pct_diff_pgd = pct_diff(stats_pgd_adv.recall_avg, stats_pgd_orig.recall_avg)
    else:
        acc_pct_diff_fgsm = 99999
        recall_pct_diff_fgsm = 99999

    if len(stats_deepfool_adv.accuracy) == 0:
        acc_pct_diff_deepfool = pct_diff(stats_deepfool_adv.accuracy_avg, stats_deepfool_orig.accuracy_avg)
        recall_pct_diff_deepfool = pct_diff(stats_deepfool_adv.recall_avg, stats_deepfool_orig.recall_avg)
    else:
        acc_pct_diff_deepfool = 99999
        recall_pct_diff_deepfool = 99999
    
    if len(stats_pgd_adv.accuracy) == 0:
        acc_pct_diff_pgd = pct_diff(stats_pgd_adv.accuracy_avg, stats_pgd_orig.accuracy_avg)
        recall_pct_diff_pgd = pct_diff(stats_pgd_adv.recall_avg, stats_pgd_orig.recall_avg)
    else:
        acc_pct_diff_pgd = 99999
        recall_pct_diff_pgd = 99999
    
    data = [
        ['FGSM', stats_fgsm_orig.recall_avg, stats_fgsm_adv.recall_avg, 
            stats_fgsm_orig.accuracy_avg, stats_fgsm_adv.accuracy_avg, recall_pct_diff_fgsm, acc_pct_diff_fgsm],
        
        ['PGD', stats_pgd_orig.recall_avg, stats_pgd_adv.recall_avg, 
            stats_pgd_orig.accuracy_avg, stats_pgd_adv.accuracy_avg, recall_pct_diff_pgd, acc_pct_diff_pgd],
        
        ['DEEPFOOL', stats_deepfool_orig.recall_avg, stats_deepfool_adv.recall_avg, 
            stats_deepfool_orig.accuracy_avg, stats_deepfool_adv.accuracy_avg, recall_pct_diff_deepfool, acc_pct_diff_deepfool]
    ]
    
    headers = [
        'ATTACK METHOD',
        f'ORIG AVG RECALL FOR {SOURCE_LABEL}',
        f'ATTACK AVG RECALL',
        'ORIG AVG ACCURACY',
        'ATTACK AVG ACCURACY',
        '% DIFF RECALL',
        '% DIFF ACCURACY'
    ]
    
    print(tabulate.tabulate(data, headers=headers), '\n')
    
    # plot data using the original predictions as the ground truth
    print('USING ORIGINAL PREDICTIONS AS GROUND TRUTH')
    data_pred_truth = [
        ['FGSM', stats_fgsm_pred.recall_avg, stats_fgsm_pred.accuracy_avg],
        ['PGD', stats_pgd_pred.recall_avg, stats_pgd_pred.accuracy_avg],
        ['DEEPFOOL', stats_deepfool_pred.recall_avg, stats_deepfool_pred.accuracy_avg]
    ]
    headers = ['ATTACK METHOD', f'AVG RECALL {SOURCE_LABEL}', f'AVG ACCURACY']
    
    print(tabulate.tabulate(data_pred_truth, headers=headers), '\n')
    
    averages = [
        ['POINTPILLAR ORIG', histo_orig_fgsm.avg_car, histo_orig_fgsm.avg_pedestrain, histo_orig_fgsm.avg_cyclist],
        ['FGSM', histo_adv_fgsm.avg_car, histo_adv_fgsm.avg_pedestrain, histo_adv_fgsm.avg_cyclist],
        ['DEEPFOOL', histo_adv_deepfool.avg_car, histo_adv_deepfool.avg_pedestrain, histo_adv_deepfool.avg_cyclist],
        ['PGD', histo_adv_pgd.avg_car, histo_adv_pgd.avg_pedestrain, histo_adv_pgd.avg_cyclist]
    ]
    headers = ['MODEL', 'CAR PROB AVG', 'PEDESTRIAN PROB AVG', 'CYCLIST PROB AVG']

    print(tabulate.tabulate(averages, headers=headers), '\n')
    
    def create_prob_histogram_adv(class_name, data_orig, data_fgsm, data_deepfool, data_pgd):
        hist = go.Figure()
        hist.add_trace(go.Histogram(
            x=data_orig,
            name='Original PointPillar',
        ))
        hist.add_trace(go.Histogram(
            x=data_fgsm,
            name='FGSM',
        ))
        hist.add_trace(go.Histogram(
            x=data_deepfool,
            name='DEEPFOOL'
        )) 
        hist.add_trace(go.Histogram(
            x=data_pgd,
            name='PGD'
        ))
        
        hist.update_layout(
            title_text = f'Adversarial {class_name} Prediction Probabilites',
            xaxis_title_text='Score',
            yaxis_title_text='Count',
            barmode='stack',
            bargroupgap=0.1
        )
        return hist


    hist_car_adv = create_prob_histogram_adv(
        'Car', 
        histo_orig_fgsm.probs_car, 
        histo_adv_fgsm.probs_car, 
        histo_adv_deepfool.probs_car, 
        histo_adv_pgd.probs_car
    )

    hist_pedestrian_adv = create_prob_histogram_adv(
        'Pedestrian', 
        histo_orig_fgsm.probs_pedestrian, 
        histo_adv_fgsm.probs_pedestrian, 
        histo_adv_deepfool.probs_pedestrian, 
        histo_adv_pgd.probs_pedestrian
    )

    hist_cyclist_adv = create_prob_histogram_adv(
        'Cyclist', 
        histo_orig_fgsm.probs_cyclist, 
        histo_adv_fgsm.probs_cyclist, 
        histo_adv_deepfool.probs_cyclist, 
        histo_adv_pgd.probs_cyclist
    )
    
    def create_stats_histogram(stat_fgsm, stat_deepfool, stat_pgd, stat_label:str):
        hist = go.Figure()
        hist.add_trace(go.Histogram(
            x=stat_fgsm,
            name='FGSM',
        ))

        hist.add_trace(go.Histogram(
            x=stat_deepfool,
            name='DEEPFOOL',
        ))

        hist.add_trace(go.Histogram(
            x=stat_pgd,
            name='PGD',
        ))

        hist.update_layout(
            title_text = f'{stat_label} With Respect to Original PointPillar Predictions',
            xaxis_title_text=stat_label,
            yaxis_title_text='Count',
        )
        return hist
        
    hist_pseudo_recall = create_stats_histogram(
        stats_fgsm_pred.recall,
        stats_deepfool_pred.recall,
        stats_pgd_pred.recall,
        f'Recall {SOURCE_LABEL}'
    )

    hist_pseudo_accuracy = create_stats_histogram(
        stats_fgsm_pred.accuracy,
        stats_deepfool_pred.accuracy,
        stats_pgd_pred.accuracy,
        f'Accuracy'
    )

    # display all the plots
    hist_car_adv.show()
    hist_pedestrian_adv.show()
    hist_cyclist_adv.show()
    hist_pseudo_recall.show()
    hist_pseudo_accuracy.show()
    
if __name__ == "__main__":
    main()
