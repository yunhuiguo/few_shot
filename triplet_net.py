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

from methods.knowledge import knowledge
from utils import *
import backbone

from datasets import svhn_few_shot, cifar_few_shot, caltech256_few_shot, ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot

#from sklearn.neighbors import KNeighborsClassifier


class Net(nn.Module):
    def __init__(self, model_func, n_way):
        super(Net, self).__init__()

        self.feature= model_func
        self.feat_dim   = self.feature.final_feat_dim
        
        self.fc = nn.Linear(self.feat_dim, n_way)

    def forward(self, x):
        x = self.feature(x)
        x = self.fc(x)
        return x

def greedy_embedding_selection(imagenet_embeddings, cifar100_embeddings, y_a_i, with_replacement=False):
        
    margin = 1.0
    TripletSelector = AllTripletSelector()
    TripletLoss = OnlineTripletLoss(margin, TripletSelector)

    embeddings_idx = []
    embeddings_set = []
    embeddings_all = None
 
    embedding_num = len(imagenet_embeddings)
    min_loss = 100.0 

    for num in range(embedding_num):
        embedding_candidate = None
        idx_candidate = -1
        for idx, embedding in enumerate(imagenet_embeddings):
            if embeddings_all is None:

                norm = embedding.norm(p=2, dim=1, keepdim=True)
                embedding_normalized =  embedding.div(norm.expand_as( embedding))
                running_loss = TripletLoss( embedding_normalized, y_a_i)
            else:   
                tmp_embedding = torch.cat((embeddings_all, embedding), 1)
                norm = tmp_embedding.norm(p=2, dim=1, keepdim=True)
                embedding_normalized =  tmp_embedding.div(norm.expand_as( tmp_embedding))
                running_loss = TripletLoss( embedding_normalized, y_a_i)

            if running_loss < min_loss:
                embedding_candidate = embedding
                idx_candidate = idx
                min_loss = running_loss

        if with_replacement:
            if idx_candidate != -1: 
                embeddings_set.append(embedding_candidate)
                embeddings_idx.append(idx_candidate)
                if embeddings_all is None:
                    embeddings_all = embedding_candidate
                else:
                    embeddings_all = torch.cat((embeddings_all, embedding_candidate), 1)
        else:
            if idx_candidate not in embeddings_idx and idx_candidate != -1: 
                embeddings_set.append(embedding_candidate)
                embeddings_idx.append(idx_candidate)
                if embeddings_all is None:
                    embeddings_all = embedding_candidate
                else:
                    embeddings_all = torch.cat((embeddings_all, embedding_candidate), 1)

    print embeddings_idx


def train_loss_cross_validation(embeddings, y_a_i, support_size, n_support, total_epoch):

    embeddings = Variable(embeddings).cuda()
    all_losses = []

    for r in range(n_support):

        train_embeddings = []
        val_embeddings = []
        train_y = []
        val_y = []

        for idx in range(embeddings.size()[0]):
            if (idx - r) % n_support == 0:
                val_embeddings.append(embeddings[idx, :].view(1, embeddings[idx, :].size()[0]))
                val_y.append(y_a_i[idx])
            else:
                train_embeddings.append(embeddings[idx, :].view(1, embeddings[idx, :].size()[0]))
                train_y.append(y_a_i[idx])

        train_y = np.asarray(train_y)
        val_y = np.asarray(val_y)

        val_embeddings = torch.cat(val_embeddings, 0)
        train_embeddings = torch.cat(train_embeddings, 0)

        loss_fn = nn.CrossEntropyLoss().cuda()
        net = Net(embeddings.size()[1]).cuda()
        classifier_opt = torch.optim.SGD(net.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
       
        train_y = Variable(torch.from_numpy(train_y)).cuda() # (25,)

        train_size = support_size - n_support
        batch_size = 4
        for epoch in range(total_epoch):
            rand_id = np.random.permutation(train_size)

            for j in range(0, train_size, batch_size):
                classifier_opt.zero_grad()

                #####################################
                selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, train_size)]).cuda()
                z_batch = train_embeddings[selected_id]

                y_batch = train_y[selected_id] 
                #####################################
                outputs = net(z_batch)            
                #####################################
        
                loss = loss_fn(outputs, y_batch)
                loss.backward()
                classifier_opt.step()

        val_y = Variable(torch.from_numpy(val_y)).cuda() # (25,)

        outputs = net(val_embeddings)   
        loss = loss_fn(outputs, val_y)
        all_losses.append(loss)

    return sum(all_losses) / (len(all_losses) + 0.0)



def train_loss_half_validation(embeddings, y_a_i, support_size, n_support, total_epoch):
    embeddings = embeddings.cpu().numpy()

    train_embeddings = []
    val_embeddings = []
    train_y = []
    val_y = []

    for idx in range(support_size):
        if (idx % 10) % 2 == 0:
            val_embeddings.append(embeddings[idx, :].reshape(1, embeddings[idx, :].shape[0]))
            val_y.append(y_a_i[idx])
        else:
            train_embeddings.append(embeddings[idx, :].reshape(1, embeddings[idx, :].shape[0]))
            train_y.append(y_a_i[idx])

    train_y = np.asarray(train_y)
    val_y = np.asarray(val_y)


    val_embeddings = torch.from_numpy(np.concatenate( val_embeddings, axis=0 ))
    train_embeddings = torch.from_numpy(np.concatenate( train_embeddings, axis=0 ))


    loss_fn = nn.CrossEntropyLoss().cuda()
    net = Net(train_embeddings.size()[1]).cuda()
    classifier_opt = torch.optim.SGD(net.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
   
    train_y = Variable(torch.from_numpy(train_y)).cuda() # (25,)
    train_embeddings = Variable(train_embeddings).cuda()


    train_size = support_size / 2
    batch_size = 4
    for epoch in range(total_epoch):
        rand_id = np.random.permutation(train_size)

        for j in range(0, train_size, batch_size):
            classifier_opt.zero_grad()

            #####################################
            selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, train_size)]).cuda()
            z_batch = train_embeddings[selected_id]

            y_batch = train_y[selected_id] 
            #####################################
            outputs = net(z_batch)            
            #####################################
    
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            classifier_opt.step()

    val_embeddings = Variable(val_embeddings).cuda()
    val_y = Variable(torch.from_numpy(val_y)).cuda() # (25,)

    outputs = net(val_embeddings)   
    loss = loss_fn(outputs, val_y)

    return loss


def combine_model(model_embeddings, y_a_i, support_size, n_support, with_replacement=True):

    embeddings_idx_model = []
    embeddings_all = None

    min_loss = 100.0 

    cross_validation_epoch = 20
    for num in range(len(model_embeddings)):
        embedding_candidate = None
        idx_candidate = -1

        for idx, embedding in enumerate(model_embeddings):

            if embeddings_all is None:
                running_loss = train_loss_half_validation(embedding, y_a_i, support_size, n_support, cross_validation_epoch)
            else:
                tmp_embedding = torch.cat((embeddings_all, embedding), 1)
                running_loss = train_loss_half_validation(tmp_embedding, y_a_i, support_size, n_support, cross_validation_epoch)

            if running_loss < min_loss:
                embedding_candidate = embedding
                idx_candidate = idx
                min_loss = running_loss

        if with_replacement:
            if idx_candidate != -1: 
                embeddings_idx_model.append(idx_candidate)
                if embeddings_all is None:
                    embeddings_all = embedding_candidate
                else:
                    embeddings_all = torch.cat((embeddings_all, embedding_candidate), 1)
        else:
            if idx_candidate not in embeddings_idx_model and idx_candidate != -1: 
                embeddings_idx_model.append(idx_candidate)
                if embeddings_all is None:
                    embeddings_all = embedding_candidate
                else:
                    embeddings_all = torch.cat((embeddings_all, embedding_candidate), 1)

    return embeddings_idx_model, embeddings_all



def train_selection(imagenet_embeddings, cifar100_embeddings, dtd_embeddings, cub_embeddings, caltech_embeddings, y_a_i, support_size, n_support, with_replacement=False):
    all_embeddings = [imagenet_embeddings, cifar100_embeddings, dtd_embeddings, cub_embeddings, caltech_embeddings]

    embeddings_idx = []

    cross_validation_epoch = 20

    embeddings_best_of_each = []
    embeddings_idx_of_each = []

    for num in range(len(all_embeddings)):
        embedding_candidate = None
        idx_candidate = -1
        min_loss = 100.0 
        for idx, embedding in enumerate(all_embeddings[num]):

            running_loss = train_loss_half_validation(embedding, y_a_i, support_size, n_support, cross_validation_epoch)
            if running_loss < min_loss:
                embedding_candidate = embedding
                idx_candidate = idx
                min_loss = running_loss
        embeddings_idx_of_each.append(idx_candidate)
        embeddings_best_of_each.append(embedding_candidate)


    embeddings_idx_model, embeddings_all = combine_model(embeddings_best_of_each, y_a_i, support_size, n_support, with_replacement=with_replacement)
   
    print(embeddings_idx_model)

    return embeddings_idx_of_each, embeddings_idx_model, embeddings_all, embeddings_best_of_each


def test_loop(novel_loader, return_std = False, loss_type="softmax", n_query = 15, n_way = 5, n_support = 5): #overwrite parrent function
    correct = 0
    count = 0

    iter_num = len(novel_loader) 

    acc_all = []
    for _, (x, y) in enumerate(novel_loader):

        ###############################################################################################

        imagenet_model = model_dict[params.model]()
        cifar100_model = model_dict[params.model]()
        cub_model = model_dict[params.model]()
        caltech256_model = model_dict[params.model]()
        dtd_model = model_dict[params.model]()
        
        ###############################################################################################      
        checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, "miniImageNet", params.model, params.method)
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

        '''
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

        #cifar100_model.load_state_dict(state)

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

        #cub_model.load_state_dict(state)
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
        '''
        n_query = x.size(1) - n_support
        x = x.cuda()
        x_var = Variable(x)

        batch_size = 4
        support_size = n_way * n_support # 25
       
        '''
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

        ###############################################################################################
        x_a_i = x_var[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:]) # (25, 3, 224, 224)
        imagenet_embeddings = []
        for idx, module in enumerate(imagenet_model.trunk):
            x_a_i = module(x_a_i)
            if len(list(x_a_i.size())) == 4:
                embedding =  F.adaptive_avg_pool2d(x_a_i, (1, 1)).squeeze()
                imagenet_embeddings.append(embedding.detach())
        imagenet_embeddings = imagenet_embeddings[4:-1]

        x_a_i = x_var[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:]) # (25, 3, 224, 224)
        cifar100_embeddings = []
        for idx, module in enumerate(cifar100_model.trunk):
            x_a_i = module(x_a_i)
            if len(list(x_a_i.size())) == 4:
                embedding =  F.adaptive_avg_pool2d(x_a_i, (1, 1)).squeeze()
                cifar100_embeddings.append(embedding.detach())
        cifar100_embeddings = cifar100_embeddings[4:-1]


        embeddings_train = imagenet_embeddings[-1]

        ##########################################################
        y_a_i = np.repeat(range( n_way ), n_support ) # (25,)


        #embeddings_idx, embeddings_train  =  train_selection(imagenet_embeddings, cifar100_embeddings, y_a_i, support_size, n_support, with_replacement=True)
        x_a_i = x_var[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:]) # (25, 3, 224, 224)
        cub_embeddings = []
        for idx, module in enumerate(cub_model.trunk):
            x_a_i = module(x_a_i)
            if len(list(x_a_i.size())) == 4:
                embedding =  F.adaptive_avg_pool2d(x_a_i, (1, 1)).squeeze()
                cub_embeddings.append(embedding.detach())
        cub_embeddings = cub_embeddings[4:-1]
    
        x_a_i = x_var[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:]) # (25, 3, 224, 224)
        caltech_embeddings = []
        for idx, module in enumerate(caltech256_model.trunk):
            x_a_i = module(x_a_i)
            if len(list(x_a_i.size())) == 4:
                embedding =  F.adaptive_avg_pool2d(x_a_i, (1, 1)).squeeze()
                caltech_embeddings.append(embedding.detach())
        caltech_embeddings = caltech_embeddings[4:-1]

        x_a_i = x_var[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:]) # (25, 3, 224, 224)
        dtd_embeddings = []
        for idx, module in enumerate(dtd_model.trunk):
            x_a_i = module(x_a_i)
            if len(list(x_a_i.size())) == 4:
                embedding =  F.adaptive_avg_pool2d(x_a_i, (1, 1)).squeeze()
                dtd_embeddings.append(embedding.detach())
        dtd_embeddings = dtd_embeddings[4:-1]

        ##########################################################
        y_a_i = np.repeat(range( n_way ), n_support ) # (25,)
        embeddings_idx_of_each, embeddings_idx_model, embeddings_train, embeddings_best_of_each = train_selection(imagenet_embeddings, cifar100_embeddings, dtd_embeddings, cub_embeddings, caltech_embeddings, y_a_i, support_size, n_support, with_replacement=True)
       
        #greedy_embedding_selection(imagenet_embeddings, cifar100_embeddings, y_a_i, False)
        ##########################################################
        x_b_i = x_var[:, n_support:,:,:,:].contiguous().view( n_way* n_query,   *x.size()[2:]) # (75, 3, 224, 224)
        imagenet_embeddings_test = []
        for idx, module in enumerate(imagenet_model.trunk):
            x_b_i = module(x_b_i)

            if len(list(x_b_i.size())) == 4:
                embedding =  F.adaptive_avg_pool2d(x_b_i, (1, 1)).squeeze()
                imagenet_embeddings_test.append(embedding)
        imagenet_embeddings_test = imagenet_embeddings_test[4:-1]
                imagenet_embeddings_test.append(embedding.detach())
        imagenet_embeddings_test = imagenet_embeddings_test[4:-1]


        x_b_i = x_var[:, n_support:,:,:,:].contiguous().view( n_way* n_query,   *x.size()[2:]) # (75, 3, 224, 224)
        cifar100_embeddings_test = []
        for idx, module in enumerate(cifar100_model.trunk):
            x_b_i = module(x_b_i)

            if len(list(x_b_i.size())) == 4:
                embedding =  F.adaptive_avg_pool2d(x_b_i, (1, 1)).squeeze()
                cifar100_embeddings_test.append(embedding.detach())     
        cifar100_embeddings_test = cifar100_embeddings_test[4:-1]
    

        for index in embeddings_idx:
            embeddings_test.append(imagenet_embeddings_test[index])

        x_b_i = x_var[:, n_support:,:,:,:].contiguous().view( n_way* n_query,   *x.size()[2:]) # (75, 3, 224, 224)
        cub_embeddings_test = []
        for idx, module in enumerate(cub_model.trunk):
            x_b_i = module(x_b_i)

            if len(list(x_b_i.size())) == 4:
                embedding =  F.adaptive_avg_pool2d(x_b_i, (1, 1)).squeeze()
                cub_embeddings_test.append(embedding.detach())
        cub_embeddings_test = cub_embeddings_test[4:-1]
        

        x_b_i = x_var[:, n_support:,:,:,:].contiguous().view( n_way* n_query,   *x.size()[2:]) # (75, 3, 224, 224)
        caltech_embeddings_test = []
        for idx, module in enumerate(caltech256_model.trunk):
            x_b_i = module(x_b_i)

            if len(list(x_b_i.size())) == 4:
                embedding =  F.adaptive_avg_pool2d(x_b_i, (1, 1)).squeeze()
                caltech_embeddings_test.append(embedding.detach())
        caltech_embeddings_test = caltech_embeddings_test[4:-1]
        

        x_b_i = x_var[:, n_support:,:,:,:].contiguous().view( n_way* n_query,   *x.size()[2:]) # (75, 3, 224, 224)
        dtd_embeddings_test = []
        for idx, module in enumerate(dtd_model.trunk):
            x_b_i = module(x_b_i)
            if len(list(x_b_i.size())) == 4:
                embedding =  F.adaptive_avg_pool2d(x_b_i, (1, 1)).squeeze()
                dtd_embeddings_test.append(embedding.detach())
        dtd_embeddings_test = dtd_embeddings_test[4:-1]

        ############################################################################################
        all_embeddings_test = [imagenet_embeddings_test, cifar100_embeddings_test, dtd_embeddings_test, cub_embeddings_test, caltech_embeddings_test]
        embeddings_test = []

        for index in embeddings_idx_model:
            embeddings_test.append(all_embeddings_test[index][embeddings_idx_of_each[index]])
      
        embeddings_test = torch.cat(embeddings_test, 1)
    
        embeddings_test = imagenet_embeddings_test[-1]
        
        ############################################################################################
        '''

        x_a_i = x_var[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:]) # (25, 3, 224, 224)
        x_b_i = x_var[:, n_support:,:,:,:].contiguous().view( n_way* n_query,   *x.size()[2:]) # (75, 3, 224, 224)

        y_a_i = Variable( torch.from_numpy( np.repeat(range( n_way ), n_support ) )).cuda() # (25,)


        net = Net(imagenet_model, n_way).cuda()
        loss_fn = nn.CrossEntropyLoss().cuda()
     
        classifier_opt = torch.optim.SGD(net.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        total_epoch = 100
        #embeddings_train = Variable(embeddings_train.cuda())

        net.train()
        for epoch in range(total_epoch):
            rand_id = np.random.permutation(support_size)

            for j in range(0, support_size, batch_size):
                classifier_opt.zero_grad()

                #####################################
                selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size)]).cuda()
                #z_batch = embeddings_train[selected_id]
                z_batch = x_a_i[selected_id]

                y_batch = y_a_i[selected_id] 
                #####################################

                outputs = net(z_batch)            
                #####################################
        
                loss = loss_fn(outputs, y_batch)
                loss.backward()
                classifier_opt.step()


        #embeddings_test = Variable(embeddings_test.cuda())
        #scores = net(embeddings_test)

        scores = net(x_b_i)

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

    image_size = 224
    optimization = 'Adam'
    n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
    iter_num = 600
    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) 
   
    dataset_names = ["ISIC", "EuroSAT", "CropDisease"]
    novel_loaders = []
    
    datamgr             =  ISIC_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(aug =False)
    novel_loaders.append(novel_loader)


    datamgr             =  EuroSAT_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(aug =False)
    novel_loaders.append(novel_loader)


    datamgr             =  CropDisease_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(aug =False)
    novel_loaders.append(novel_loader)
    
    #########################################################################

    for idx, novel_loader in enumerate(novel_loaders):
        print dataset_names[idx]
        start_epoch = params.start_epoch
        stop_epoch = params.stop_epoch

        test_loop(novel_loader, return_std = False,  n_query = 15, **few_shot_params)
