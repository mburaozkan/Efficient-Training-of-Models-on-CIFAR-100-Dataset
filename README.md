# Efficient-Training-of-Models-on-CIFAR-100-Dataset

## Requirements
Project is created with:
* Python version: 3.10
* torch  version: 2.1.0.dev20230608+cu121,

## Usage

1. Enter the directory
```
$ cd folder_name
```

2. Train the model
```
$ python train.py -net googlenet -gpu
```

3. Test the model
```
$ python test.py -net googlenet -gpu -weights path_to_googlenet_weights_file
```

## Copyrights
The code template was taken from https://github.com/weiaicunzai/pytorch-cifar100.
