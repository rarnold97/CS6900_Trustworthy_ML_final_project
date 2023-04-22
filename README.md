# Overview
This project implements three adversarial attack methods on Lidar point clouds,
using the Pointpillar model provided by the OpenPCDet Team [OpenPCDet Github](Pytorch with CUDA in WSL2 Documentation.md).  The following attack methods are explored:

1. Fast Gradient Sign Method (FGSM): [FGSM Source Code](./final_project/attack.py)
2. Projected Gradient Descent Method (PGD): [PGD Source Code](./final_project/attack.py)
3. Deepfool Method: [Deepfool Source Code](./final_project/attack.py)

# Requirements
I did all of my development in WSL2 Ubuntu 18, using an anaconda environment with python 3.8.  I have provided an Anaconda [environment.yml](./environment.yml) file that captured the libraries I had success with.  Before attempting to install it, refer to the following prep steps:

- Ensure you have an OpenPCDet checkpoint, stored in `./checkpoints`
    - I provided the following Pointpillar checkpoint file: [pointpillar_7728.pth](./checkpoints/pointpillar_7728.pth)

- download OpenPCDet by cloning submodules:
    - `git submodule update --init --recursive`

- Visit NVIDIA's website to download CUDA drivers for WSL.  I operated with CUDA 11.8. [NVIDIA CUDA Link](https://developer.nvidia.com/cuda-downloads).

- Install the spconv image processing guide.  An example using CUDA 11.8:
    - `$ pip install spconv-cu118`

- Install Pytorch 1.10 (this is provided in the environment.yml file)

- Download the Kitti Dataset (Defer to the [OpenPCDet Instructions](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md) on where to place downloads) 

- [KITTI Dataset Download Page](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)

For a more in-depth guide on how to setup CUDA to effectively work with WSL2, Refer to [pytorch_with_CUDA_wsl.md](./pytorch_with_CUDA_wsl.md)

# (Optional) Install the final project to your environment

I provided a [pyproject.toml](./pyproject.toml) poetry file to allow developers to
install the package to their virtual environment, to make python pathing easier.
- `$ poetry install`

# Basic Usage

The main entry point to this project is in the [main.py](./main.py) script, which heavily uses the final_project package, pcdet, and pytorch.  A sample demo file was provided as a Jupyter notebook: [Pre-Generated Jupyter DEMO](./demo.ipynb).

The run the main entrypoint (assuming all dependencies have been setup and resolved):

```bash
$ python main.py
```