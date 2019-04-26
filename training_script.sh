#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 nohup python ./train.py --dataset cifar100_to_caltech256 --model ResNet18 --method baseline --train_aug > ./training_logs/cifar100_to_caltech256_baseline.log &
CUDA_VISIBLE_DEVICES=0 nohup python ./train.py --dataset caltech256_to_cifar100 --model ResNet18 --method baseline --train_aug > ./training_logs/caltech256_to_cifar100_baseline.log &


CUDA_VISIBLE_DEVICES=1 nohup python ./train.py --dataset miniImageNet_to_CUB --model ResNet18 --method baseline --train_aug > ./training_logs/miniImageNet_to_CUB_baseline.log &
CUDA_VISIBLE_DEVICES=1 nohup python ./train.py --dataset CUB_to_miniImageNet --model ResNet18 --method baseline --train_aug > ./training_logs/CUB_to_miniImageNet_baseline.log &

CUDA_VISIBLE_DEVICES=2 nohup python ./train.py --dataset omniglot_to_emnist --model Conv4 --method baseline > ./training_logs/omniglot_to_emnist_baseline.log &



python ./train.py --dataset omniglot_to_emnist --model Conv4 --method baseline