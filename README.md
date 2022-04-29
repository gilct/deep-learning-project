# Deep Learning Project (Spring 2022)

Repository for the project of EE-559 Deep Learning.

## Authors
- Ali Benabdallah
- João Correia
- Gil Tinde

## Setup

### Execution Environment

(Same as course VM)

```
conda create -n deeplearning python=3.9.7
conda activate deeplearning
conda install -c pytorch pytorch=1.9.0 torchvision=0.10.0
conda install -c anaconda jupyter
conda install -c conda-forge tqdm
pip install alive-progress
```

Could be some package errors, can help to simply use default versions of torch
```
conda install -c pytorch pytorch torchvision
```