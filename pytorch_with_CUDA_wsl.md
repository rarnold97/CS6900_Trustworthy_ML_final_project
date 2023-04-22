
# Pre-Requisites

Installing CUDA in WSL requires extra care, since it is not technically native Linux.  In additional complication is that PyTorch is not yet synced up with the latest CUDA runtime environment.  As of: `03/22/2023` , the latest CUDA toolkit version is: `12.1`.  The latest supported cuda toolkit version by PyTorch is: `11.8`. 

**Disclaimer:** If you are using a newer NVIDIA graphics card (i.e., 3000 series-ish and above), I highly recommend installing the latest cuda toolkit version supported by PyTorch.  My developer machine contains a: `NVIDIA RTX A2000 8GB Laptop GPU`, which is comparable to an `NVIDIA RTX 3050-Ti`.  I tested the `10.2` CUDA toollkit, which was too out of date for my GPU.
[Reference NVIDIA Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl)

## NVIDIA Drivers

1. Head to: [NVIDIA Driver Download](https://www.nvidia.com/Download/index.aspx) and select the correct OS and GPU Specs, specific to your machine containing an NVIDIA GPU.  These drivers will allow you to access the NVIDIA GPU through WSL2.
2. Download the installer, and follow the prompts accordingly
3. Once everything is installed, restart your machine, and ensure you have the latestes windows updates.
4. Open a wsl terminal session
5. Verify successful installation by running: `$ nvidia-smi`
6. If done correctly, you will get coherent output for the above command

## CUDA Installation Instructions

1. Head to : [NVIDIA CUDA v11.8 Archived Download](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=runfile_local) and follow the instructions for downloading the driver.  At the time of writing, 11.8 was the suitable version.  In the future, this may need to be changed to the current and latest supported version by PyTorch.
2. Select the proper steps in the isntallation interface widget buttons.  In my case, this involved: `>Linux>x86_64 > WSL-Ubuntu > 2.0 > runfile (local)` 
	- I recommend using the runfile, because it will give a prompt for a more customized installation, and put the dependencies in the proper locations
3. The forum will give instructions on how to install the runfile properly.  I will also outline this below:
	- head to the opt drive, to keep install files organized: `cd /opt`
	- `$ sudo wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run`
	- `$ sudo sh cuda_11.8.0_520.61.05_linux.run`
4. follow the prompts, and only install the CUDA toolkit.  DO NOT install the drivers, the NVIDIA drivers are managed by the host windows system.  Uncheck anything that is not the toolkit option
5. Install the cuDNN libraries
	- head to [NVIDIA cuDNN download](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
	- install zlib: `$ sudo apt-get install zlib1g`
	- Go to: NVIDIA cuDNN home page.
	- Click Download.
	- Complete the short survey and click Submit.
	- Accept the Terms and Conditions. A list of available download versions of cuDNN displays.
	- Select the cuDNN version that you want to install. A list of available resources displays.
	- Navigate to your `<cudnn>` directory containing the cuDNN tar file.  (I recommend downloading/copying this file to `/opt`)
	- Before issuing the following commands, you must replace X.Y and v8.x.x.x with your specific CUDA and cuDNN versions and package date.
		- `$ sudo tar -xvf cudnn-linux-x86_64-8.x.x.x_cudaX.Y-archive.tar.xz`
		- `$ sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda-X.Y/include `
		- `$ sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda-X.Y/lib64`
		- `$ sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*`
6. Modify your .bashrc profile to conatain the following (*Note:* That my versioning may be different than yours):
```bash
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.8
```
7. Test CUDA by running the following:
	- `$ nvcc --version`
	- If installed properly, you should get output that indicates so.
	- Example:
	- `nvcc: NVIDIA (R) Cuda compiler driver
		`Copyright (c) 2005-2022 NVIDIA Corporation
		`Built on Wed_Sep_21_10:33:58_PDT_2022
		`Cuda compilation tools, release 11.8, V11.8.89
		`Build cuda_11.8.r11.8/compiler.31833905_0`

## Anaconda and PyTorch Setup
1. Create a new anaconda environment by issuing the following (*Note, name the env to whatever you desire.  I will use pytorch for clarity sake*)
	- `$ conda create --name pytorch`
2. Activate the environment: `conda activate pytorch`
	- this will put in parantheses: `(pytorch)` in front of your shell commamd line.
3. Since we used the CUDA v11.8 toolkit, we are going to install that.  Also note, I stub out a specific version of pytorch, since my application required it.  I included this, in case this applies to you:
	- `$ conda install pytorch=1.10 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
4. Validate that torch and cuda work by issuing the following in a python session:
```bash
$ python
>> import torch
>> torch.cuda.is_available()
>> torch.cuda.get_device_name(0)
```
5. Make sure that the above steps execute without error, and return `True` and the correct device specs, according to your system's GPU

## Installing ONCE Benchmark Dataset Models

The models were designed in the following repo: [ONCE Dataset Models](https://github.com/PointsCoder/ONCE_Benchmark).

[Original Installation Documentation](https://github.com/PointsCoder/ONCE_Benchmark/blob/master/docs/INSTALL.md).

1. Install the provided requirements file: `$pip install -r requirements.txt`
2. the spec for cv2 will not install through the requirements file, and needs to be installed by the following command:
	- `$ pip install opencv-python`
3. Install the `spconv` library.  The original documentation recommends `v1.x` , which is deppreciated now. 
	- head to [spconv PYPI](https://pypi.org/project/spconv/), and follow their instructions.  I will outline what I had to do for the CUDA 11.8 toolkit:
	- `$ pip install spconv-cu118`
4. Navigate to the root of the ONCE Benchmark repo and run to compile CUDA operators:
	- `$ python setup.py develop`
5. (Optional): compile CUDA operators for DCN (Deformable Convs)
	- `$ cd pcdet/ops/dcn`
	- `$ python setup.py develop`
