import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file  

from datasets import svhn_few_shot, cifar_few_shot, caltech256_few_shot, ISIC_few_shot


#from utils import load_pretrained_model

def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params):    
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
       raise ValueError('Unknown optimization, please define by yourself')

    max_acc = 0       

    for epoch in range(start_epoch,stop_epoch):
        model.train()
        model.train_loop(epoch, base_loader,  optimizer ) #model are called by reference, no need to return 
        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        acc = model.test_loop( val_loader)
        if acc > max_acc : #for baseline and baseline++, we don't use validation here so we let acc = -1
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

    return model

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')

    if params.dataset not in ['cifar100_to_cifar10', 'caltech256_to_cifar100']:

        if params.dataset == 'miniImageNet_to_ISIC':
            base_file = configs.data_dir['miniImagenet'] + 'all.json' 
            val_file   = configs.data_dir['CUB'] + 'val.json' 

        elif params.dataset == 'CUB_to_miniImageNet':
            base_file = configs.data_dir['CUB'] + 'base.json' 
            val_file   = configs.data_dir['miniImagenet'] + 'val.json' 

        elif params.dataset == 'omniglot_to_emnist':
            base_file = configs.data_dir['omniglot'] + 'noLatin.json' 
            val_file   = configs.data_dir['emnist'] + 'val.json' 

    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'omniglot_to_emnist']:
            image_size = 28
        else:
            image_size = 84
    else:
        image_size = 224

    if params.dataset in ['omniglot', 'omniglot_to_emnist']:
        assert params.model == 'Conv4' and not params.train_aug ,'omniglot only support Conv4 without augmentation'
        params.model = 'Conv4S'


    optimization = 'Adam'
    if params.method in ['baseline', 'baseline++'] :

        if params.dataset not in ["caltech256_to_cifar100", "cifar100_to_caltech256"]:

            base_datamgr    = SimpleDataManager(image_size, batch_size = 16)
            base_loader     = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
            
            val_datamgr     = SimpleDataManager(image_size, batch_size = 64)
            val_loader      = val_datamgr.get_data_loader( val_file, aug = False)
        
        elif params.dataset == "cifar100_to_cifar10":
            base_datamgr    = cifar_few_shot.SimpleDataManager(224, "CIFAR100", batch_size = 16)
            base_loader    = base_datamgr.get_data_loader( "base" , aug = True )

            val_datamgr     = caltech256_few_shot.SimpleDataManager(224, "CIFAR10", batch_size = 64)
            val_loader      = val_datamgr.get_data_loader( 'val', aug = False)
            
        elif params.dataset == "caltech256_to_cifar100":
            base_datamgr    = caltech256_few_shot.SimpleDataManager(image_size, batch_size = 16)
            base_loader     = base_datamgr.get_data_loader( "base" , aug = True )

            val_datamgr     = cifar_few_shot.SimpleDataManager(image_size, batch_size = 64)
            val_loader      = val_datamgr.get_data_loader( 'val', aug = False)


        if params.dataset == 'caltech256_to_cifar100':
            params.num_classes = 256
            assert params.num_classes >= 256, 'class number need to be larger than max label id in base class'

        if params.dataset == 'omniglot':
            assert params.num_classes >= 4112, 'class number need to be larger than max label id in base class'
        
        if params.dataset == 'omniglot_to_emnist':
            params.num_classes = 1597
            assert params.num_classes >= 1597, 'class number need to be larger than max label id in base class'


        if params.method == 'baseline':
            model           = BaselineTrain( model_dict[params.model], params.num_classes)
            #net = load_pretrained_model.load_multi_branch_pretrained_model(resume, 50, )
            #model           = BaselineTrain( net, params.num_classes)
        
        elif params.method == 'baseline++':
            model           = BaselineTrain( model_dict[params.model], params.num_classes, loss_type = 'dist')


    elif params.method in ['protonet','matchingnet','relationnet', 'relationnet_softmax', 'maml', 'maml_approx']:
        n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
        test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot) 


        if params.dataset not in ["caltech256_to_cifar100", "cifar100_to_caltech256"]:
            base_datamgr            = SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
            base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
            val_datamgr             = SetDataManager(image_size, n_query = n_query, **test_few_shot_params)
            val_loader              = val_datamgr.get_data_loader( val_file, aug = False) 

        elif params.dataset == "cifar100_to_caltech256":
             
            base_datamgr            = cifar_few_shot.SetDataManager('base', image_size, n_query = n_query, **train_few_shot_params)
            base_loader             = base_datamgr.get_data_loader(aug = params.train_aug)
           
            val_datamgr             = caltech256_few_shot.SetDataManager('val', image_size, n_query = n_query, **test_few_shot_params)
            val_loader              = val_datamgr.get_data_loader( aug = False) 
       
        elif params.dataset == "caltech256_to_cifar100":
             
            base_datamgr            = caltech256_few_shot.SetDataManager('base', image_size, n_query = n_query, **train_few_shot_params)
            base_loader             = base_datamgr.get_data_loader(aug = params.train_aug)
          
            val_datamgr             = cifar_few_shot.SetDataManager('val', image_size, n_query = n_query, **test_few_shot_params)
            val_loader              = val_datamgr.get_data_loader( aug = False) 

        if params.method == 'protonet':
            model           = ProtoNet( model_dict[params.model], **train_few_shot_params )
        elif params.method == 'matchingnet':
            model           = MatchingNet( model_dict[params.model], **train_few_shot_params )
        elif params.method in ['relationnet', 'relationnet_softmax']:
            if params.model == 'Conv4': 
                feature_model = backbone.Conv4NP
            elif params.model == 'Conv6': 
                feature_model = backbone.Conv6NP
            elif params.model == 'Conv4S': 
                feature_model = backbone.Conv4SNP
            else:
                feature_model = lambda: model_dict[params.model]( flatten = False )
            loss_type = 'mse' if params.method == 'relationnet' else 'softmax'

            model           = RelationNet( feature_model, loss_type = loss_type , **train_few_shot_params )
        elif params.method in ['maml' , 'maml_approx']:
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            backbone.ResNet.maml = True
            model           = MAML(  model_dict[params.model], approx = (params.method == 'maml_approx') , **train_few_shot_params )
            if params.dataset in ['omniglot', 'omniglot_to_emnist']: #maml use different parameter in omniglot
                model.n_task     = 32
                model.task_update_num = 1
                model.train_lr = 0.1
    else:
       raise ValueError('Unknown method')


    model = model.cuda()

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if not params.method  in ['baseline', 'baseline++']: 
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch


    model = train(base_loader, val_loader,  model, optimization, start_epoch, stop_epoch, params)