import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
from itertools import combinations

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML

from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file 

from methods.featurenet_cifar import featurenet_cifar
from methods.knowledge import knowledge
from utils import *
import backbone

from datasets import svhn_few_shot, cifar_few_shot, caltech256_few_shot, ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot


class Net(nn.Module):
    def __init__(self, dim):
        super(Net, self).__init__()
        self.fc = nn.Linear(dim, 5)

    def forward(self, x):
        x = self.fc(x)
        return x

def test_loop(novel_loader, return_std = False, loss_type="softmax", n_query = 15, n_way = 5, n_support = 5): #overwrite parrent function
    correct = 0
    count = 0

    iter_num = len(novel_loader) 

    acc_all = []

    for _, (x, _) in enumerate(novel_loader):

        start = time.time()
        ###############################################################################################

        imagenet_model = model_dict[params.model]()
        cifar100_model = model_dict[params.model]()
        cub_model = model_dict[params.model]()
        caltech256_model = model_dict[params.model]()
        dtd_model = model_dict[params.model]()

        ###############################################################################################      
        checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, "miniImageNet_to_CUB", params.model, params.method)
        if params.train_aug:
            checkpoint_dir += '_aug'

        params.save_iter = -1
        if params.save_iter != -1:
            imagenet_modelfile   = get_assigned_file(checkpoint_dir, params.save_iter)
        elif params.method in ['baseline', 'baseline++'] :
            imagenet_modelfile   = get_resume_file(checkpoint_dir)
        else:
            imagenet_modelfile   = get_best_file(checkpoint_dir)


        tmp = torch.load(imagenet_modelfile)
        state = tmp['state']

        state_keys = list(state.keys())
        for _, key in enumerate(state_keys):
            if "feature." in key:
                newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                state[newkey] = state.pop(key)
            else:
                state.pop(key)

        imagenet_model.load_state_dict(state)
        ###############################################################################################      

        
        ###############################################################################################      
        checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, "cifar100", params.model, params.method)
        if params.train_aug:
            checkpoint_dir += '_aug'

        params.save_iter = -1
        if params.save_iter != -1:
            cifar100_modelfile   = get_assigned_file(checkpoint_dir, params.save_iter)
        elif params.method in ['baseline', 'baseline++'] :
            cifar100_modelfile   = get_resume_file(checkpoint_dir)
        else:
            cifar100_modelfile   = get_best_file(checkpoint_dir)


        tmp = torch.load(cifar100_modelfile)
        state = tmp['state']

        state_keys = list(state.keys())
        for _, key in enumerate(state_keys):
            if "feature." in key:
                newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                state[newkey] = state.pop(key)
            else:
                state.pop(key)

        cifar100_model.load_state_dict(state)
        ###############################################################################################
        
        ###############################################################################################      
        checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, "CUB", params.model, params.method)
        if params.train_aug:
            checkpoint_dir += '_aug'

        params.save_iter = -1
        if params.save_iter != -1:
            cub_modelfile   = get_assigned_file(checkpoint_dir, params.save_iter)
        elif params.method in ['baseline', 'baseline++'] :
            cub_modelfile   = get_resume_file(checkpoint_dir)
        else:
            cub_modelfile   = get_best_file(checkpoint_dir)

        tmp = torch.load(cub_modelfile)
        state = tmp['state']

        state_keys = list(state.keys())
        for _, key in enumerate(state_keys):
            if "feature." in key:
                newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                state[newkey] = state.pop(key)
            else:
                state.pop(key)

        cub_model.load_state_dict(state)
        ###############################################################################################
        checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, "caltech256", params.model, params.method)
        if params.train_aug:
            checkpoint_dir += '_aug'

        params.save_iter = -1
        if params.save_iter != -1:
            caltech_modelfile   = get_assigned_file(checkpoint_dir, params.save_iter)
        elif params.method in ['baseline', 'baseline++'] :
            caltech_modelfile   = get_resume_file(checkpoint_dir)
        else:
            caltech_modelfile   = get_best_file(checkpoint_dir)

        tmp = torch.load(caltech_modelfile)
        state = tmp['state']

        state_keys = list(state.keys())
        for _, key in enumerate(state_keys):
            if "feature." in key:
                newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                state[newkey] = state.pop(key)
            else:
                state.pop(key)

        caltech256_model.load_state_dict(state)
        ###############################################################################################
        checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, "DTD", params.model, params.method)
        if params.train_aug:
            checkpoint_dir += '_aug'

        params.save_iter = -1
        if params.save_iter != -1:
            dtd_modelfile   = get_assigned_file(checkpoint_dir, params.save_iter)
        elif params.method in ['baseline', 'baseline++'] :
            dtd_modelfile   = get_resume_file(checkpoint_dir)
        else:
            dtd_modelfile   = get_best_file(checkpoint_dir)

        tmp = torch.load(dtd_modelfile)
        state = tmp['state']

        state_keys = list(state.keys())
        for _, key in enumerate(state_keys):
            if "feature." in key:
                newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                state[newkey] = state.pop(key)
            else:
                state.pop(key)

        dtd_model.load_state_dict(state)
        ###############################################################################################

        n_query = x.size(1) - n_support
        x = x.cuda()
        x_var = Variable(x)

        batch_size = 4
        support_size = n_way * n_support # 25
    
        ###############################################################################################
        imagenet_model.cuda()
        cifar100_model.cuda()
        cub_model.cuda()
        caltech256_model.cuda()
        dtd_model.cuda()

        imagenet_model.eval()
        cifar100_model.eval()
        cub_model.eval()
        caltech256_model.eval()
        dtd_model.eval()

        x_a_i = x_var[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:]) # (25, 3, 224, 224)
        imagenet_embeddings = []
        for idx, module in enumerate(imagenet_model.trunk):
            x_a_i = module(x_a_i)

            if len(list(x_a_i.size())) == 4:
                embedding =  F.adaptive_avg_pool2d(x_a_i, (1, 1)).squeeze()
                imagenet_embeddings.append(embedding)

        imagenet_embeddings = imagenet_embeddings[4:-1]

        
        x_a_i = x_var[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:]) # (25, 3, 224, 224)
        cifar100_embeddings = []
        for idx, module in enumerate(cifar100_model.trunk):
            x_a_i = module(x_a_i)
            if len(list(x_a_i.size())) == 4:
                embedding =  F.adaptive_avg_pool2d(x_a_i, (1, 1)).squeeze()
                cifar100_embeddings.append(embedding)

        cifar100_embeddings = cifar100_embeddings[4:-1]
        

        x_a_i = x_var[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:]) # (25, 3, 224, 224)
        cub_embeddings = []
        for idx, module in enumerate(cub_model.trunk):
            x_a_i = module(x_a_i)
            if len(list(x_a_i.size())) == 4:
                embedding =  F.adaptive_avg_pool2d(x_a_i, (1, 1)).squeeze()
                cub_embeddings.append(embedding)

        cub_embeddings = cub_embeddings[4:-1]
    

        x_a_i = x_var[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:]) # (25, 3, 224, 224)
        caltech_embeddings = []
        for idx, module in enumerate(caltech256_model.trunk):
            x_a_i = module(x_a_i)
            if len(list(x_a_i.size())) == 4:
                embedding =  F.adaptive_avg_pool2d(x_a_i, (1, 1)).squeeze()
                caltech_embeddings.append(embedding)

        caltech_embeddings = caltech_embeddings[4:-1]


        x_a_i = x_var[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:]) # (25, 3, 224, 224)
        dtd_embeddings = []
        for idx, module in enumerate(dtd_model.trunk):
            x_a_i = module(x_a_i)
            if len(list(x_a_i.size())) == 4:
                embedding =  F.adaptive_avg_pool2d(x_a_i, (1, 1)).squeeze()
                dtd_embeddings.append(embedding)

        dtd_embeddings = dtd_embeddings[4:-1]

        embeddings_train = dtd_embeddings[-2]
        ###############################################################################################

        x_b_i = x_var[:, n_support:,:,:,:].contiguous().view( n_way* n_query,   *x.size()[2:]) # (75, 3, 224, 224)
        imagenet_embeddings_test = []
        for idx, module in enumerate(imagenet_model.trunk):
            x_b_i = module(x_b_i)

            if len(list(x_b_i.size())) == 4:
                embedding =  F.adaptive_avg_pool2d(x_b_i, (1, 1)).squeeze()
                imagenet_embeddings_test.append(embedding)
      
        imagenet_embeddings_test = imagenet_embeddings_test[4:-1]

    
    
        x_b_i = x_var[:, n_support:,:,:,:].contiguous().view( n_way* n_query,   *x.size()[2:]) # (75, 3, 224, 224)
        cifar100_embeddings_test = []
        for idx, module in enumerate(cifar100_model.trunk):
            x_b_i = module(x_b_i)

            if len(list(x_b_i.size())) == 4:
                embedding =  F.adaptive_avg_pool2d(x_b_i, (1, 1)).squeeze()
                cifar100_embeddings_test.append(embedding)
      
        cifar100_embeddings_test = cifar100_embeddings_test[4:-1]

        
        x_b_i = x_var[:, n_support:,:,:,:].contiguous().view( n_way* n_query,   *x.size()[2:]) # (75, 3, 224, 224)
        cub_embeddings_test = []
        for idx, module in enumerate(cub_model.trunk):
            x_b_i = module(x_b_i)

            if len(list(x_b_i.size())) == 4:
                embedding =  F.adaptive_avg_pool2d(x_b_i, (1, 1)).squeeze()
                cub_embeddings_test.append(embedding)
      
        cub_embeddings_test = cub_embeddings_test[4:-1]
        

        x_b_i = x_var[:, n_support:,:,:,:].contiguous().view( n_way* n_query,   *x.size()[2:]) # (75, 3, 224, 224)
        caltech_embeddings_test = []
        for idx, module in enumerate(caltech256_model.trunk):
            x_b_i = module(x_b_i)

            if len(list(x_b_i.size())) == 4:
                embedding =  F.adaptive_avg_pool2d(x_b_i, (1, 1)).squeeze()
                caltech_embeddings_test.append(embedding)
      
        caltech_embeddings_test = caltech_embeddings_test[4:-1]
        

        x_b_i = x_var[:, n_support:,:,:,:].contiguous().view( n_way* n_query,   *x.size()[2:]) # (75, 3, 224, 224)
        dtd_embeddings_test = []
        for idx, module in enumerate(dtd_model.trunk):
            x_b_i = module(x_b_i)

            if len(list(x_b_i.size())) == 4:
                embedding =  F.adaptive_avg_pool2d(x_b_i, (1, 1)).squeeze()
                dtd_embeddings_test.append(embedding)
      
        dtd_embeddings_test = dtd_embeddings_test[4:-1]

        embeddings_test =  dtd_embeddings_test[-2]

        ###############################################################################################

        net = Net(embeddings_test.size()[1]).cuda()
        loss_fn = nn.CrossEntropyLoss().cuda()
        classifier_opt = torch.optim.SGD(net.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
        total_epoch = 100

        embeddings_train = Variable(embeddings_train.cuda())
        y_a_i = Variable( torch.from_numpy( np.repeat(range( n_way ), n_support ) )).cuda() # (25,)

        for epoch in range(total_epoch):
            rand_id = np.random.permutation(support_size)

            for j in range(0, support_size, batch_size):
                classifier_opt.zero_grad()

                #####################################
                selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size)]).cuda()
                z_batch = embeddings_train[selected_id]

                y_batch = y_a_i[selected_id] 
                #####################################

                outputs = net(z_batch)            
                #####################################
        
                loss = loss_fn(outputs, y_batch)
                loss.backward()
                classifier_opt.step()


        embeddings_test = Variable(embeddings_test.cuda())
        scores = net(embeddings_test)

        y_query = np.repeat(range( n_way ), n_query )
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        correct_this, count_this = float(top1_correct), len(y_query)
        print (correct_this/ count_this *100)
        acc_all.append((correct_this/ count_this *100))

        ###############################################################################################

    acc_all  = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
    
if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')

    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'omniglot_to_emnist']:
            image_size = 28
        else:
            image_size = 84
    else:
        image_size = 224


    n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small

    if params.method in ['baseline']:
        iter_num = 600
        few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) 

        datamgr             =  EuroSAT_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
        novel_loader        = datamgr.get_data_loader(aug =False)

    #########################################################################

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch


    test_loop(novel_loader, return_std = False,  n_query = 15, **few_shot_params)
