# Embedding-based Music Emotion Recognition Using Composite Loss (EMER-CL)

**This repository was originally published on https://mu-lab.info/naoki_takashima/emer-cl and has been transferred here because of the of the last author.**

## Build environment
Please build the execution environment according to your own environment.

The following three points are described as samples of environment construction.
- docker (recommended) <[Link](#docker-environment)>
- Anaconda <[Link](#anaconda-environment)>
- venv <[Link](#venv-environment)>


### Dependent versions
The following environments have been verified to work.
In particular, TensorFlow is strongly dependent on the Python version.
```sh
python=3.6.x
tensorflow=1.15.x
```

### Required to be installed on the host OS
We use [FFmpeg](https://www.ffmpeg.org) to process music.

If you don't use docker, you need to install FFmpeg on the host machine.
In addition, if you are on MacOS or Ubuntu, you can download it using the following method.

```
# MacOS
brew install ffmpeg

# Ubuntu
apt install ffmpeg
```

For other operating systems, please refer to [Download FFmpeg](https://www.ffmpeg.org/download.html).

### docker environment
#### docker setup
For docker installation, please refer to the official documentation [Get Docker](https://docs.docker.com/get-docker/).
You also need to install [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker#nvidia-container-toolkit) if you want to run on GPU environment.

#### Environment setup
#### Procedure for creating an environment.
The docker commands are wrapped in a Makefile.
So, if you have a docker environment in place, please use the following commands to create your environment.
Please check the Makefile for details.

```sh
# create image and container
## CPU environment
make up
## GPU environment
make up-gpu

# Check whether the required modules for python have been installed
make log

# and then attach created container
make attach

# container stop and remove
make down
```

### anaconda environment

#### Anaconda setup
For information on installing Anaconda, please refer to the official documentation at [Installation](https://docs.anaconda.com/anaconda/install/index.html#installation) or Miniconda's [Installation Guide](https://docs.conda.io/en/latest/miniconda.html#miniconda) for more information.


#### Environment setup
If you are using Anaconda to build your environment, you can easily build your environment by using a pre-made YAML file.

```sh
# Please set the path of this repository to `WORKDIR` environment variable
export WORKDIR=$EMER_CL
## for CPU environment
conda env create -f $WORKDIR/env/conda/cpu.yml
## for GPU environment
conda env create -f $WORKDIR/env/conda/gpu.yml

# activation
conda activate EMER_CL
# deactivation
conda deactivate

# remove environment
conda remove -n EMER_CL --all
```


### venv environment

You can build the EMER-CL environment with `venv` only if your Python version is `3.6.x`.

#### Environment setup

```sh
# Please set the path of this repository to `WORKDIR` environment variable
export WORKDIR=$EMER_CL
python3 -m venv $WORKDIR/.venv
# activation
source $WORKDIR/.venv/bin/activate

# pip upgrade
pip install -U pip
# install other libraries
## for CPU environment
pip install -r $WORKDIR/env/venv/cpu.txt
## for GPU environment
pip install -r $WORKDIR/env/venv/gpu.txt

# deactivation
deactivate

# remove environment
rm -rf $WORKDIR/.venv
```

## Usage

## Pre-processing of dataset
Download the DEAM dataset and the PMEmo dataset and format them to the data format used by EMER-CL.
See `$WORKDIR/src/data/README.md` for details.

## Train and evaluation
You can run `train.py` after [Build environment](#build-environment) is ready.
Please check `$WORKDIR/src/README.md` for details.

Our hyper-parameter tuning of EMER-CL is provided in [this supplemental document](https://drive.google.com/file/d/1JXyNZBBMegX_OD7DUa_PPSuipAIfjZ-8/view?usp=sharing).

## Detailed analysis
If the training completes, the trained model will be saved in `$WORKDIR/src/model`. This can then be used to perform detailed analysis.

See `$WORKDIR/src/analysis/README.md` for details.

## Reference

If you found this code useful, please cite the following paper:

```
@article{takashima2021crossmodal,
    title={Embedding-based Music Emotion Recognition Using Composite Loss},
    author={Takashima, Naoki and Li, Frédéric and Grzegorzek, Marcin and Shirahama, Kimiaki},
    journal={IEEE Access},
    volume={11},
    pages={36579-36604},
    year={2023}
}
```

Details of hyper-parameter tuning and expriments are provided in [this supplemental document](https://drive.google.com/file/d/1JXyNZBBMegX_OD7DUa_PPSuipAIfjZ-8/view?usp=sharing).

## License

[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
