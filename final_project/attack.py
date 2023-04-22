from dataclasses import dataclass
from pathlib import Path
import typing
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torchvision.transforms as transforms
import torchvision.models as models
from torch import Tensor

import numpy as np
import copy
from PIL import Image

from final_project.utils import OriginalData, get_encoded_cls_label
from pcdet.models.detectors.pointpillar import PointPillar
from pcdet.models.model_utils import model_nms_utils

# from git submodule
from cw_attack import cw

__all__ = [
    "AttackParams",
    "ParamsDeepFool",
    "ParamsPGD",
    "ParamsFGSM",
    "ParamsCW",
    "AdvAttack"
]


#progress_bar = tqdm.tqdm(total=len(pretrain_model_params.data_loader), 
#                         leave=True, desc='Classify Data', dynamic_ncols=True)
#progress_bar.update()
#progress_bar.close()

@dataclass
class AttackParams:
    #cfg.CLASS_NAMES: ['Car', 'Pedestrian', 'Cyclist']
    class_names: typing.List = None
    num_classes: int = 3
    max_iters: int = 100
    epsilon: float = 0.3
    positive_only_features: bool = True
    clamp: Tuple[float, float] = (-1.0, 1.0)
    label: Tensor = None

    @classmethod
    def from_file(cls, filename: Path):
        pass


@dataclass
class ParamsDeepFool(AttackParams):
    overshoot = 0.02


@dataclass
class ParamsPGD(AttackParams):
    step_size: float = 1e-3
    loss_function: _Loss = None


@dataclass
class ParamsCW(AttackParams):
    is_targeted: bool = False
    confidence: float = 0.0
    c_range: typing.Tuple[float, float] = (1e-3, 1e10)
    search_steps: int = 5
    abort_early: bool = True
    box: typing.Tuple[float, float] = (-1., 1.)
    optimizer_lr: float = 1e-2
    init_rand: bool = False


# perform adversarial attacks and generate adversarial examples

class AdvAttack:

    @staticmethod
    def deepfool_toy(image: torch.tensor, model: nn.Module, num_classes: int=10, overshoot: float =0.02, max_iter:int=10):
        """
        Copy of the example given from Medium.com, authored by Aminul Huq.
        :link https://medium.com/@aminul.huq11/pytorch-implementation-of-deepfool-53e889486ed4
        source code: https://github.com/aminul-huq/DeepFool/tree/master
        """
        
        f_image = model.forward(image).data.numpy().flatten()
        I = (np.array(f_image)).flatten().argsort()[::-1]
        
        I = I[0:num_classes]
        label = I[0]
        
        input_shape = image.detach().numpy().shape
        pert_image = copy.deepcopy(image)
        w = np.zeros(input_shape)
        r_tot = np.zeros(input_shape)
        
        loop_i = 0
        
        x = torch.tensor(pert_image[None, :], requires_grad=True)
        
        fs: Tensor = model.forward(x[0])
        k_i = label
        
        while k_i == label and loop_i < max_iter:
            
            pert = np.inf
            fs[0, I[0]].backward(retain_graph=True)
            grad_orig = x.grad.data.numpy().copy()
            
            for k in range(1, num_classes):
                fs[0, I[k]].backward(retain_graph=True)
                cur_grad = x.grad.data.numpy().copy()
                
                # set new w_k and f_k
                w_k = cur_grad - grad_orig
                f_k = (fs[0, I[k]] - fs[0, I[0]]).data.numpy()
                
                pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())
                
                # determine which w_k to use
                if pert_k < pert:
                    pert = pert_k
                    w = w_k
            # compute r_i and r_tot
            # Added 1e-4 for numerical stability
            r_i = (pert+1e-4) * w / np.linalg.norm(w)
            r_tot = np.float32(r_tot + r_i)
            
            pert_image = image + (1 + overshoot)*torch.from_numpy(r_tot)
            
            x = torch.tensor(pert_image, requires_grad=True)
            fs = model.forward(x[0])
            k_i = np.argmax(fs.data.numpy().flatten())
            
            loop_i += 1
            
        r_tot = (1+overshoot)*r_tot
        
        return r_tot, loop_i, label, k_i, pert_image

    @staticmethod
    def deepfool_pcdet(data: OriginalData, model: nn.Module, params: ParamsDeepFool):
        """
        Copy of the example given from Medium.com, authored by Aminul Huq.
        :link https://medium.com/@aminul.huq11/pytorch-implementation-of-deepfool-53e889486ed4
        source code: https://github.com/aminul-huq/DeepFool/tree/master

        Not electing to use GPU, Therefore numpy is a better CPU choice.
        """
        def predictor(model: PointPillar, class_names: typing.List, X: Tensor, gt_boxes: Tensor, batch_size: int = 1):

            data_dict = {
                'spatial_features_2d': X.clone().cuda().requires_grad_(True),
                'gt_boxes': gt_boxes.cuda().requires_grad_(True),
                'batch_size': batch_size
            }
            #with torch.no_grad():
            # derrived from model.post_processing method
            # we need to glean the probabilities fpr deepfool
            dense_output = model.dense_head.forward(data_dict)
            cls_scores = torch.squeeze(dense_output['batch_cls_preds'])
            cls_probs = torch.sigmoid(cls_scores)
            cls_preds, raw_labels = torch.max(cls_probs, dim=-1)
            selected, _ = model_nms_utils.class_agnostic_nms(
                box_scores=cls_preds,
                box_preds=torch.squeeze(dense_output['batch_box_preds']),
                nms_config=model.model_cfg.POST_PROCESSING.NMS_CONFIG,
                score_thresh=model.model_cfg.POST_PROCESSING.SCORE_THRESH
            )
            pred_labels = raw_labels + 1
            pred_labels = pred_labels[selected]
            pred_dict, _ = model.post_processing(dense_output)
            for probs in cls_probs[selected, :]:
                probs[0].backward(retain_graph=True)
            
            for key, val in pred_dict[0].items():
                if torch.is_tensor(val):
                    pred_dict[0][key] = val.cpu()
                    
            torch.cuda.empty_cache()
            return data_dict['spatial_features_2d'], pred_labels, cls_probs[selected, :].clone().requires_grad_(True), pred_dict[0]
        
        #f_x = model.forward(X).data.numpy().flatten()
        pert_output, prediction, cls_probs, output_dict = predictor(
            model,
            params.class_names,
            data.feature_data, 
            data.batch_data['gt_boxes']
        )
        
        orig_label = prediction.cpu()
        input_shape = data.feature_data.detach().numpy().shape

        w = np.zeros(input_shape)
        r_tot = np.zeros(input_shape)

        loop_i = 0
        current_pred = prediction.cpu()
        for idx in range(orig_label.size()[0]):
            loop_i = 0
            print(f'PROCESSING: OBJECT AT: {idx}')
            target_label = orig_label[idx]
            if len(current_pred) != len(orig_label):
                break
            while target_label == orig_label[idx] and loop_i < params.max_iters:
                print(f"--> processing deepfool iteration: {loop_i+1} with original label: {orig_label[idx]}")
                pert = np.inf
                grad_orig = pert_output.grad.data.cpu().numpy().copy()
                
                for k in range(1, len(params.class_names)):
                    cls_probs[idx][k].backward(retain_graph=True)
                    cur_grad = pert_output.grad.data.cpu().numpy().copy()
                    
                    # set new w_k and f_k
                    w_k = cur_grad - grad_orig
                    f_k = (cls_probs[idx][k] - cls_probs[idx][0]).data.cpu().numpy()
                    
                    pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())
                    
                    # determine which w_k to use
                    if pert_k < pert:
                        pert = pert_k
                        w = w_k
                # compute r_i and r_tot
                # Added 1e-4 for numerical stability
                r_i = (pert+1e-4) * w / np.linalg.norm(w)
                r_tot = np.float32(r_tot + r_i)
                
                pert_output = data.feature_data + (1 + params.overshoot)*torch.from_numpy(r_tot).to(data.feature_data.device)    
                pert_output, current_pred, cls_probs, output_dict = predictor(
                    model, 
                    params.class_names, 
                    pert_output, 
                    data.batch_data['gt_boxes']
                )
                
                current_pred = current_pred.cpu()
                
                loop_i += 1
                if loop_i == params.max_iters:
                    print("DEEPFOOL MAXIMUM ITERATIONS MET !!!")

                if len(orig_label) != len(current_pred):
                    break
                
                target_label = current_pred[idx]
                
                
            r_tot = (1+params.overshoot)*r_tot

        print("===================================================")
        print(f"DEEP FOOL TOOK: {loop_i} iterations to converge")
        print(f"ORIGINAL LABEL: {orig_label}")
        print(f"NEW LABEL: {target_label}")
        print("===================================================")
        torch.cuda.empty_cache()
        pert_output_cpu = pert_output.clone().detach().cpu()
        del pert_output
        for key, val in output_dict.items():
            if torch.is_tensor(val):
                output_dict[key] = val.detach().cpu()
                
        return pert_output_cpu, output_dict

    @staticmethod
    def cw2_untargeted(X: Tensor, model: nn.Module, params: ParamsCW)->Tensor:
        adversary = cw.L2Adversary(
            targeted=params.is_targeted,
            confidence=params.confidence,
            c_range=params.c_range,
            search_steps=params.search_steps,
            max_steps=params.max_iters,
            abort_early=params.abort_early,
            box=params.box,
            optimizer_lr=params.optimizer_lr,
            init_rand=params.init_rand
        )
        
        return adversary(model, X, params.labels, to_numpy=False)
         
    # TODO adapt this chat GPT example to be compatible in a general case, or at the very least, with the ONCE dataset. 
    @staticmethod
    def pgd_attack(X, model: nn.Module, params: ParamsPGD):
        """
        Projected Gradient Descent Attack.

        Parameters
        ----------
        X : Tensor
            Input feature data.
        params : ParamsPGD
            defines the attack method parameters, as outlined in the class definition.
            These include - model, input feature tensor (X), labels, epsilon for tolerance,
            and the loss function
        """

        x_adv = X.clone().detach().requires_grad_(True).to(X.device)
        targeted = params.target_label is not None
        if not targeted:
            params.target_label = params.model_predictor(X, model)
            
        num_channels: int = X.size()[1]
        _x_adv = x_adv.clone().detach()

        for _ in range(params.max_iters):
            probs = model.forward(x_adv)
            probs.flatten().argsort(descending=True)
            # get rid of thebatch dimension and backward propagate from the 
            # node with highest probability
            for i in range(params.num_classes):
                torch.squeeze(probs)[i].backward(retain_graph=True)

            gradients = x_adv.grad.data
            
            if targeted:
                # Targeted attack: Gradient descent with on the loss of the (incorrect) target label
                # w.r.t the input data
                _x_adv -= gradients.sign() * params.step_size
            else:
                # Untargeted: Gradient ascent on the loss of the correct label
                # w.r.t. the model parameters
                _x_adv += gradients.sign() * params.epsilon
                    
            _x_adv = torch.max(torch.min(_x_adv, X + params.epsilon), X - params.epsilon) 
            _x_adv = _x_adv.clamp(*(params.clamp))
            
            x_adv.data = _x_adv.data.clone()
            current_label = params.model_predictor(_x_adv, model)
            print(current_label)
            
            # terminate early if we have breached the decision boundary
            if targeted:
                if current_label == params.target_label:
                    break
            else:
                if current_label != params.target_label:
                    break
        
        return x_adv.detach()
    
    @staticmethod
    def pgd_attack_pcdet(data: OriginalData, model: nn.Module, params: ParamsPGD):
        
        def predictor(model: PointPillar, class_names: typing.List, X: Tensor, gt_boxes: Tensor, batch_size: int = 1):

            data_dict = {
                'spatial_features_2d': X.cuda().requires_grad_(True),
                'gt_boxes': gt_boxes.cuda().requires_grad_(True),
                'batch_size': batch_size
            }
            
            #with torch.no_grad():
            dense_output = model.dense_head.forward(data_dict)
            
            pred_dicts, _ = model.post_processing(dense_output)
            n = pred_dicts[0]['pred_scores'].size()[0]
            for i in range(n):
                pred_dicts[0]['pred_scores'][i].backward(retain_graph=True)
            labels = []
            for box_dict in pred_dicts:
                pred_labels = box_dict['pred_labels'].cpu().numpy()
                labels.append(get_encoded_cls_label(
                    {'name': np.array(class_names)[pred_labels - 1]}
                ))

            torch.cuda.empty_cache()
            return labels, pred_dicts       


        x_adv = data.feature_data.data.clone().detach().requires_grad_(True)
        gt_boxes = data.batch_data['gt_boxes']
        
        print('~~~~~ PGD ATTACK STATUS ~~~~~')
        
        for step in range(params.max_iters):
            _x_adv = x_adv.clone().detach().requires_grad_(True)
            
            # assuming batch size of 1
            batch_prediction, batch_pred_dict = predictor(model, params.class_names, _x_adv, gt_boxes)
            prediction = batch_prediction[0]  #assuming batch size of 1
            prediction_dict = batch_pred_dict[0]
            print(f'--> STEP: {step+1} PREDICTIONS: | {" | ".join([str(val.item()) for val in prediction])}|')
                    
            # TODO look into using provided loss function instead
            with torch.no_grad():
                gradients = _x_adv.grad.sign() * params.step_size
                x_adv += gradients

            x_adv = torch.max(torch.min(x_adv, data.feature_data + params.epsilon), data.feature_data - params.epsilon)
            x_adv = x_adv.clamp(*(params.clamp))
            
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        return x_adv.detach().to('cpu'), prediction_dict
                
        
    # FGSM attack code
    @staticmethod
    def fgsm_attack_pcdet(data: OriginalData, model: nn.Module, params: AttackParams):

        def predictor(model: PointPillar, class_names: typing.List, X: Tensor, gt_boxes: Tensor, batch_size: int = 1):

            data_dict = {
                'spatial_features_2d': X.cuda().requires_grad_(True),
                'gt_boxes': gt_boxes.cuda().requires_grad_(True),
                'batch_size': batch_size
            }
            
            #with torch.no_grad():
            dense_output = model.dense_head.forward(data_dict)
            
            pred_dicts, label = model.post_processing(dense_output)
            n = pred_dicts[0]['pred_scores'].size()[0]
            for i in range(n):
                pred_dicts[0]['pred_scores'][i].backward(retain_graph=True)

            box_dict = pred_dicts[0]
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            label = get_encoded_cls_label(
                {'name': np.array(class_names)[pred_labels - 1]}
            )
            
            torch.cuda.empty_cache()

            return label, box_dict

        print('--> PERFORMING FGSM ATTACK')
        # Collect the element-wise sign of the data gradient
        x_adv = data.feature_data.clone().detach().requires_grad_(True).to(data.feature_data.device)
        orig_label, output_dict = predictor(model, params.class_names, x_adv, data.batch_data['gt_boxes'])
        sign_data_grad = x_adv.grad.data.sign()

        orig_label = orig_label.cpu()
        print(f'--> ORIGINAL LABEL: | {" | ".join([str(element.item()) for element in orig_label])}')
        
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_data = data.feature_data + params.epsilon*sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_data = torch.clamp(perturbed_data, params.clamp[0], params.clamp[1])
        # Return the perturbed image

        new_label, output_dict = predictor(model, params.class_names, perturbed_data, data.batch_data['gt_boxes'])
        print(f'--> PERTURBED LABEL: | {" | ".join([str(element.item()) for element in new_label])}')
        
        for key, val in output_dict.items():
            if torch.is_tensor(val):
                output_dict[key] = val.detach().cpu()
                
        return perturbed_data.detach().cpu(), output_dict
        
def load_test_data()->Tuple[Tensor, nn.Module]:
    model: nn.Module = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    # set to evaluate mode, rather than training mode
    model.eval()

    # load sample imnage from resnet dataset, and construct a tensor
    im_original = Image.open('./toy_data/test_im1.jpg')
    # standardize
    img = transforms.Compose([transforms.ToTensor(),])(im_original)
    
    # add extra dimension (usually the batch dimension)
    img = img[None, :, :, :].clone().detach().requires_grad_(True)
    return img, model

"""
Test function for determining if attack works as intended.
Importing resnet as a pretrained model to test algorithm.
"""
def test_deepfool_attack():

    img, net = load_test_data()
    params_deepfool = ParamsDeepFool(num_classes=10, max_iters=50)
    perturbed_image, new_label, old_label = AdvAttack.deepfool(img, net, params_deepfool)

def test_cw_attack():
    
    img, net = load_test_data()
    cw_params = ParamsCW(clamp=(0.0, 1.0))
    # 88 is the original label value
    single_label = torch.tensor([88,])
    
    adv_example = AdvAttack.cw2_untargeted(img.clone().detach(), net, single_label, cw_params)
    def classify(X: Tensor):
        return torch.argmax(net.forward(X).flatten()).item()
    
    adv_predict = classify(adv_example)
    orig_predict = classify(img)
    
    print('~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('__CW ATTACK RESULTS__')
    print(f'ORIGINAL LABEL: {orig_predict}')
    print(f'ATTACKED LABEL {adv_predict}')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~')
    
    assert adv_predict != orig_predict, \
        'Same Classification label indicates Deepfool attack failed ...'
        
def test_pgd_attack():
    img, net = load_test_data()
    pgd_params = ParamsPGD(clamp=(0.0, 1.0), step_size=0.00001, epsilon=0.3, max_iters=1000, \
        model_predictor=lambda x, model: torch.argmax(model(x).flatten()))
     
    adv_predict = AdvAttack.pgd_attack(X=img, params=pgd_params, model=net, \
                                       model_predictor=lambda x, model: torch.argmax(model(x).flatten()))

    new_label = torch.argmax(net(adv_predict).flatten())
    
    print('~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('__CW ATTACK RESULTS__')
    print(f'ORIGINAL LABEL: {88}')
    print(f'ATTACKED LABEL {new_label}')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~')
    
    assert new_label != 88, \
        'Same classification label indicates PGD attack failed ...'
    
def main():
    #test_deepfool_attack() 
    #test_cw_attack()
    #test_pgd_attack()
    pass
 
if __name__ == "__main__":
    main()