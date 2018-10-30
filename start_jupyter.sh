#!/usr/bin/env bash

PYTHONPATH=PYTHONPATH:`pwd`/cifar10 CUDA_VISIBLE_DEVICES=0 jupyter-notebook --no-browser --ip 0.0.0.0 --port 9000 --NotebookApp.token=""  --NotebookApp.password=""