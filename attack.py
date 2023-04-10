from dataclasses import dataclass
from pathlib import Path
import typing

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
import copy
from PIL import Image

# from git submodule
from cw_attack import cw


@dataclass
class AttackParams:
    model: nn.Module
    num_classes: int = 5
    max_iters: int = 100
    epsilon: float = 0.3
    positive_only_features: bool = True

    @classmethod
    def from_file(cls, filename: Path):
        pass


@dataclass
class ParamsDeepFool(AttackParams):
    overshoot = 0.02


@dataclass
class ParamsPGD(AttackParams):
    loss_function: _Loss = None
    clamp: typing.Tuple[float, float] = (0.0, 1.0)
    target_labels: torch.Tensor = None
    labels: torch.Tensor
    step_size: float = 1e-3
    

@dataclass
class ParamsFGSM(AttackParams):
    clamp: typing.Tuple[float, float] = (0.0, 1.0)


@dataclass
class ParamsCW(AttackParams):
    is_targeted: bool = False
    confidence: float = 0.0
    c_range: typing.Tuple[float, float] = (1e-3, 1e10)
    search_steps: int = 5
    abort_early: bool = True
    box: typing.Tuple[float, float] = (-1., 1.)
    optimizer_lr: float = 1e-2,
    init_rand: bool = False


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
        
        fs: torch.Tensor = model.forward(x[0])
        fs_list = [fs[0, I[k]] for k in range(num_classes)]
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
    def cw2_untargeted(X: torch.Tensor, labels: torch.Tensor, params: ParamsCW):
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
        
        return adversary(params.model, X, labels, to_numpy=False)
         
    @staticmethod
    def deepfool(X: torch.Tensor, params: ParamsDeepFool):
        """
        Implementation of the DeepFool Adversarial attack

        Parameters
        ----------
        X : torch.tensor
            Feature Tensor
        net : nn.Module
            _description_
        num_classes : int
            _description_
        overshoot : float, optional
            _description_, by default 0.02
        max_iter : int, optional
            _description_, by default 100
        """

        f_output_layer: torch.Tensor = torch.flatten(params.model.forward(X))
        predict_probs: torch.Tensor = None
        
        if max(f_output_layer.size()) > params.num_classes:
            f_output_layer = f_output_layer.argsort(descending=True)
            predict_probs = f_output_layer[0:params.num_classes]
        else:
            predict_probs = f_output_layer

        label = predict_probs[0]
        
        input_shape = X.detach().size()
        perturbed_output = X.detach().clone()
        w = torch.zeros(input_shape)
        r_total = torch.zeros(input_shape)
        
        loop_ctr: int = 0
        
        x = perturbed_output[None, :].detach().clone().requires_grad_(True)        
        
        fs: torch.Tensor = params.model.forward(x[0])
        k_i = label
        
        while k_i == label and loop_ctr < params.max_iters:
            
            pertubation = torch.inf
            fs[0, predict_probs[0]].backward(retain_graph=True)
            original_gradient = x.grad.data.detach().clone()
        
            for k in range(1, params.num_classes):
                fs[0, predict_probs[k]].backward(retain_graph=True)
                current_gradient = x.grad.data
            
                # set new w_k and f_k
                w_k = current_gradient - original_gradient
                f_k = fs[0, predict_probs[k]] - fs[0, predict_probs[0]]

                pertubation_k = f_k.abs() / torch.linalg.norm(w_k.flatten())
            
                # determine which w_k to use
                if pertubation_k < pertubation:
                    pertubation = pertubation_k
                    w = w_k
            # compute r_i and r_tot
            # Added 1e-4 for numerical stability
            r_i = (pertubation+1e-4) * w / torch.linalg.norm(w)
            r_total = (r_total + r_i).to(torch.float32)
            
            perturbed_output = X + (1.0 + params.overshoot)*r_total
            x = perturbed_output.detach().clone().requires_grad_(True)        
            fs = params.model.forward(x[0])
            k_i = (torch.flatten(fs)).argmax().item()
        
            loop_ctr += 1
            
        r_total = (1.0 + params.overshoot)*r_total
    
        print(f"Deep Fool Attack took {loop_ctr} iterations to complete")
        print(f"Deepfool Label vector used: {label}")

        #return r_tot, loop_i, label, k_i, pert_image
        return perturbed_output

    # TODO adapt this chat GPT example to be compatible in a general case, or at the very least, with the ONCE dataset. 
    @staticmethod
    def pgd_attack(X: torch.Tensor, labels: torch.Tensor, params: ParamsPGD):
        """
        Projected Gradient Descent Attack.

        Parameters
        ----------
        X : torch.Tensor
            Input feature data.
        params : ParamsPGD
            defines the attack method parameters, as outlined in the class definition.
            These include - model, input feature tensor (X), labels, epsilon for tolerance,
            and the loss function
        """
        x_adv = X.clone().detach().requires_grad_(True).to(X.device)
        targeted = params.target_labels is not None
        num_channels: int = X.shape[1]
        
        for _ in range(params.max_iters):
            _x_adv = x_adv.clone().detach().requires_grad_(True)
            
            prediction: torch.Tensor = params.model(_x_adv)
            loss = params.loss_function(prediction, params.target_labels if targeted else params.labels)
            loss.backward()
            
            # using no-gradient context, since we are doing our own gradient computations
            with torch.no_grad():
                # Force the gradient step to be a fixed size in a certain norm
                gradients = _x_adv.grad * params.step_size / _x_adv.grad.view(_x_adv.shape[0], -1)\
                    .norm(dim=-1).view(-1, num_channels, 1, 1)
                        
                if targeted:
                    # Targeted attack: Gradient descent with on the loss of the (incorrect) target label
                    # w.r.t the input data
                    x_adv -= gradients
                else:
                    # Untargeted: Gradient ascent on the loss of the correct label
                    # w.r.t. the model parameters
                    x_adv += gradients
                    
            # Project back into l_norm ball and correct range
            delta = x_adv - X
            
            # Assume x and x_adv are batched tensors where the first dimension
            # is a batched dimension
            mask = delta.view(delta.shape[0], -1).norm(dim=1) <= params.epsilon
            scaling_factor = delta.view(delta.shape[0], -1).norm(dim=1)
            scaling_factor[mask] = params.epsilon
            
            # .view() assumes batched images as a 4D Tensor
            delta *= params.epsilon / scaling_factor.view(-1, 1, 1, 1)
            
            x_adv = X + delta
            
            x_adv = x_adv.clamp(*(params.clamp))

            current_labels = params.model(x_adv)
            
            # terminate early if we have breached the decision boundary
            if targeted:
                if torch.all(current_labels == params.target_labels):
                    break
            else:
                if torch.all(params.labels != current_labels):
                    break
        
        return x_adv.detach()
    
    
    # FGSM attack code
    @staticmethod
    def fgsm_attack(X: torch.Tensor, params: ParamsFGSM):
        # Collect the element-wise sign of the data gradient
        data_grad = X.detach().grad
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_data = X + params.epsilon*sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_data = torch.clamp(perturbed_data, params.clamp[0], params.clamp[10])
        # Return the perturbed image
        return perturbed_data


"""
Test function for determining if attack works as intended.
Importing resnet as a pretrained model to test algorithm.
"""
def test_deepfool_attack(use_toy: bool = True):
    # download the resnet pre-trained model
    net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    net.eval()  # switch model to evaluation mode
    
    im_orig = Image.open('./toy_data/test_im1.jpg')  # open toy data of a bird
    # remove the mean
    img = transforms.Compose([
        transforms.ToTensor()
    ])(im_orig)
    
    # add an extra dimension for weight tensor
    img  = img[None, :, :, :]

    if use_toy:
        r, iters, label_orig, label_pert, pert_image = AdvAttack.deepfool_toy(img, net, max_iter=100)
        print(f'number of iterations performed: {iters}\nORIGINAL LABEL: {label_orig}\nPERTURBED LABEL: {label_pert}')
    else:
        pert_image = AdvAttack.deepfool(img, net, 10)
        
def main():
    #test_deepfool_attack(use_toy=True)
    test_deepfool_attack(use_toy=False) 
if __name__ == "__main__":
    main()