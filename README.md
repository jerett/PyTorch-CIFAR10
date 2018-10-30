# PyTorch for CIFAR10
This project demonstrates some personal examples with PyTorch on CIFAR10 dataset.

 ---
 
 
## Introduction

---
The CIFAR10 dataset is 32x32 size, 50000 train images and 10000 test images.
The dataset is divided into 40000 train images, 10000 validation images, and 10000 images.


## Features

* Test for many models, each model is a a little different from orgin for 
32*32 input, and will contiune to add new model.
* [Visdom](https://github.com/facebookresearch/visdom) realtime visualization of loss, acc, port 8097.
* Use [torchnet](https://github.com/pytorch/tnt) for training.
* Use jupyter book for recording echo model training process.

## Train
* Run visdom first, python -m visdom.server & 
* Open the jupyter file for the corresponding model, and then run all cells.


## Requirements
* torch
* torchvision
* numpy
* torchnet
* visdom


## Result
All result is tested on 10000 test images.You can lookup the jupyter for more details.


 Model | Accuracy
 :---: | :---: 
 [SVM](linear_classifier.ipynb) | 34.27% | 
 [Softmax](linear_classifier.ipynb) | 35.67% |
 [small-ResNet20](small_resnet20.ipynb) | 91.38%
 [small-ResNet32](small_resnet32.ipynb) | 92.53%
 [small-ResNet56](small_resnet56.ipynb) | 93.31%
 [vgg11](vgg13.ipynb) | 91.25%
 [vgg13](vgg13.ipynb) | 92.84%
 [vgg16](vgg16.ipynb) | 92.94%%
 [MobileNetV1](mobilenet_v1.ipynb) | 92.45%
 [MobileNetV2](mobilenet_v2.ipynb) | 92.47%
