from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import sys

import pytorch_resnet_branch, pytorch_resnet

from collections import OrderedDict


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def get_multi_branch_model(layers, num_classes, pretrained=True):

    if layers == 18:
        net = pytorch_resnet_branch.resnet18(pretrained, num_classes)
    elif layers == 34:
        net = pytorch_resnet_branch.resnet34(pretrained, num_classes)
    elif layers == 50:
        net = pytorch_resnet_branch.resnet50(pretrained, num_classes)
    elif layers == 101:
        net = pytorch_resnet_branch.resnet101(pretrained, num_classes)

    return net

def load_multi_branch_pretrained_model(resume, layers, num_classes, load_fc=False):
  
    net_old = get_multi_branch_model(layers, num_classes, pretrained=False)

    checkpoint = torch.load(resume)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']

    net_old = torch.nn.DataParallel(net_old).cuda()
    net_old.load_state_dict(checkpoint['state_dict'])

    net = get_multi_branch_model(layers, num_classes, pretrained=False)

    # load pretrained net 
    store_data = []
    for name, m in net_old.named_modules():
        if isinstance(m, nn.Conv2d):
            store_data.append(m.weight.data)

    element = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data = torch.nn.Parameter(store_data[element])
            element += 1


    store_data = []
    store_data_bias = []
    store_data_rm = []
    store_data_rv = []
    for name, m in net_old.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            store_data.append(m.weight.data)
            store_data_bias.append(m.bias.data)
            store_data_rm.append(m.running_mean)
            store_data_rv.append(m.running_var)

    element = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
                m.weight.data = torch.nn.Parameter(store_data[element].clone())
                m.bias.data = torch.nn.Parameter(store_data_bias[element].clone())
                m.running_var = store_data_rv[element].clone()
                m.running_mean = store_data_rm[element].clone()
                element += 1

    if load_fc is True:
        store_fc_weights = []
        store_fc_bias = []
        for name, m in net_old.named_modules():
            if isinstance(m, nn.Linear):
                store_fc_weights.append(m.weight.data)
                store_fc_bias.append(m.bias.data)

        element = 0
        for name, m in net.named_modules():
            if isinstance(m, nn.Linear):
                    m.weight.data = torch.nn.Parameter(store_fc_weights[element].clone())
                    m.bias.data = torch.nn.Parameter(store_fc_bias[element].clone())
                    element += 1
    else:
        for idx, _ in enumerate(net.fcs):
            net.fcs[idx] = Identity()

    return net


def get_model(layers, num_classes, pretrained=True):

    if layers == 18:
        net = pytorch_resnet.resnet18(pretrained, num_classes)
    elif layers == 34:
        net = pytorch_resnet.resnet34(pretrained, num_classes)
    elif layers == 50:
        net = pytorch_resnet.resnet50(pretrained, num_classes)
    elif layers == 101:
        net = pytorch_resnet.resnet101(pretrained, num_classes)

    return net


def load_pretrained_model(layers, num_classes, load_fc=False):

    net_old = get_model(layers, num_classes=1000, pretrained=True)
    net = get_model(layers, num_classes, pretrained=False)

      # load pretrained net 
    store_data = []
    for name, m in net_old.named_modules():
        if isinstance(m, nn.Conv2d):
            store_data.append(m.weight.data)

    element = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data = torch.nn.Parameter(store_data[element])
            element += 1

    store_data = []
    store_data_bias = []
    store_data_rm = []
    store_data_rv = []
    for name, m in net_old.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            store_data.append(m.weight.data)
            store_data_bias.append(m.bias.data)
            store_data_rm.append(m.running_mean)
            store_data_rv.append(m.running_var)

    element = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
                m.weight.data = torch.nn.Parameter(store_data[element].clone())
                m.bias.data = torch.nn.Parameter(store_data_bias[element].clone())
                m.running_var = store_data_rv[element].clone()
                m.running_mean = store_data_rm[element].clone()
                element += 1

    if load_fc is True:
    	net.linear.weight.data = torch.nn.Parameter(net_old.module.linear.weight.data)
    	net.linear.bias.data = torch.nn.Parameter(net_old.module.linear.bias.data)  

    else:
        net.fc = Identity()

    return net


def load_trained_model(layers, num_classes, load_fc=False, resume=None):

    net_old = get_model(layers, num_classes=5, pretrained=False)
    net = get_model(layers, num_classes, pretrained=False)

    checkpoint = torch.load(resume)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']

    net_old = torch.nn.DataParallel(net_old).cuda()
    net_old.load_state_dict(checkpoint['state_dict'])


      # load pretrained net 
    store_data = []
    for name, m in net_old.named_modules():
        if isinstance(m, nn.Conv2d):
            store_data.append(m.weight.data)

    element = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data = torch.nn.Parameter(store_data[element])
            element += 1

    store_data = []
    store_data_bias = []
    store_data_rm = []
    store_data_rv = []
    for name, m in net_old.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            store_data.append(m.weight.data)
            store_data_bias.append(m.bias.data)
            store_data_rm.append(m.running_mean)
            store_data_rv.append(m.running_var)

    element = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
                m.weight.data = torch.nn.Parameter(store_data[element].clone())
                m.bias.data = torch.nn.Parameter(store_data_bias[element].clone())
                m.running_var = store_data_rv[element].clone()
                m.running_mean = store_data_rm[element].clone()
                element += 1

    if load_fc is True:
        net.linear.weight.data = torch.nn.Parameter(net_old.module.linear.weight.data)
        net.linear.bias.data = torch.nn.Parameter(net_old.module.linear.bias.data)  
    else:
        net.fc = Identity()

    return net