#!/usr/bin/env bash

#CUDA_VISIBLE_DEVICES=0 nohup python ./train.py --dataset miniImageNet --model ResNet10 --method protonet --train_aug > ./training_logs/proto_logs.log &

#CUDA_VISIBLE_DEVICES=0 nohup python ./train.py --dataset miniImageNet --model ResNet10 --method matchingnet --train_aug > ./training_logs/MatchingNet_logs.log &

CUDA_VISIBLE_DEVICES=1 nohup python ./train.py --dataset miniImageNet --model ResNet10 --method maml_approx --train_aug > ./training_logs/MAML_logs.log &

#CUDA_VISIBLE_DEVICES=1 nohup python ./train.py --dataset miniImageNet --model ResNet10 --method relationnet  --train_aug > ./training_logs/RelationNet_logs.log &


