# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

import os
import h5py
import data.feature_loader as feat_loader
import random

class knowledge(MetaTemplate):
    def __init__(self, model_func,  n_way=5, n_support=5, loss_type = 'softmax'):
        super(knowledge, self).__init__( model_func,  n_way, n_support, change_way = False)

        self.feature    = model_func()
        self.loss_type = loss_type

        self.n_task     = 4
        self.task_update_num = 5

        self.loss_fn = nn.CrossEntropyLoss().cuda()

        self.n_way = n_way
        self.n_support = n_support

    def forward(self,x):

        out  = self.feature.forward(x)

        return out

    def set_forward(self, x, is_feature = False):
        assert is_feature == False, 'MAML do not support fixed feature' 
        x = x.cuda()
        x_var = Variable(x)

        x_a_i = x_var[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[2:]) 
        x_b_i = x_var[:,self.n_support:,:,:,:].contiguous().view( self.n_way* self.n_query,   *x.size()[2:]) 

        y_a_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_support ) )).cuda()
        batch_size = 4
        support_size = self.n_way * self.n_support

        if self.loss_type == 'softmax':
            self.classifier = nn.Linear(self.feature.final_feat_dim, self.n_way).cuda()
            self.classifier.bias.data.fill_(0)
        
        elif self.loss_type == 'dist': #Baseline ++
            self.classifier = backbone.distLinear(self.feature.final_feat_dim, self.n_way).cuda()

        set_optimizer = torch.optim.SGD(self.classifier.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
        feature_optimizer = torch.optim.Adam(self.feature.parameters(), lr = 0.01)

        for epoch in range(100):
            support_size = self.n_way * self.n_support
            rand_id = np.random.permutation(support_size)

            for i in range(0, support_size , batch_size):
                set_optimizer.zero_grad()
                #feature_optimizer.zero_grad()

                selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()
                z_batch = x_a_i[selected_id]
                y_batch = y_a_i[selected_id] 
                scores = self.forward(z_batch)
                '''
                if epoch % 10 == 0:
                    topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
                    topk_ind = topk_labels.cpu().numpy()
                    top1_correct = np.sum(topk_ind[:,0] == y_batch.cpu().numpy())
                    print float(top1_correct) / len(y_batch) * 100
                '''
                loss = self.loss_fn(scores, y_batch)
                loss.backward()
                set_optimizer.step()
                #feature_optimizer.step()

        scores = self.forward(x_b_i)
        return scores

    def set_forward_adaptation(self,x, is_feature = False): #overwrite parrent function
        raise ValueError('MAML performs further adapation simply by increasing task_upate_num')

    def set_forward_loss(self, x):
        scores = self.set_forward(x, is_feature = False)
        y_b_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_query   ) )).cuda()
        loss = self.loss_fn(scores, y_b_i)
        return loss

    def train_loop(self, epoch, train_loader, optimizer, val_loader, novel_loader, novel_loader_save,  cifar_base_loader ): #overwrite parrent function
        print_freq = 10

        avg_loss=0
        task_count = 0
        loss_all = []

        ####################################################            
        novel_file = os.path.join(os.getcwd() + "/tmp",  "novel_cifar_" + str(epoch)+ ".hdf5") 
        self.save_features(novel_loader_save , novel_file)
        cl_data_file = feat_loader.init_loader(novel_file)
        iter_num = 100
        acc_all = []
        for i in range(iter_num):
            acc = self.feature_evaluation(cl_data_file, n_way = 5, n_support = 5, n_query = 15, adaptation = False)
            acc_all.append(acc)

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        ####################################################            

        feature_optimizer = torch.optim.Adam(self.feature.parameters(), lr = 0.01)
        #feature_optimizer =  torch.optim.SGD(self.feature.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
        feature_optimizer.zero_grad()

        #for i, (x, _) in enumerate(train_loader):
        for i, (x, _) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support

            loss = self.set_forward_loss(x)
            
            avg_loss = avg_loss+loss.item()
            loss_all.append(loss)
            task_count += 1

            if task_count == self.n_task:
                loss_q = torch.stack(loss_all).sum(0)
                #loss_q.backward()
                #feature_optimizer.step()
                task_count = 0
                loss_all = []

            #feature_optimizer.zero_grad()
            
            if i % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))
    
                ####################################################            
                novel_file = os.path.join(os.getcwd() + "/tmp",  "novel_cifar_" + str(epoch)+ ".hdf5") 
                self.save_features(novel_loader_save , novel_file)
                cl_data_file = feat_loader.init_loader(novel_file)
                
                iter_num = 100
                acc_all = []
                for i in range(iter_num):
                    acc = self.feature_evaluation(cl_data_file, n_way = 5, n_support = 5, n_query = 15, adaptation = False)
                    acc_all.append(acc)

                acc_all  = np.asarray(acc_all)
                acc_mean = np.mean(acc_all)
                acc_std  = np.std(acc_all)
                print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
                ######################################################

    def save_features(self, data_loader, outfile):
        f = h5py.File(outfile, 'w')
        max_count = len(data_loader)*data_loader.batch_size
        all_labels = f.create_dataset('all_labels',(max_count,), dtype='i')
        all_feats=None
        count=0
        for i, (x,y) in enumerate(data_loader):
            if i%10 == 0:
                print('{:d}/{:d}'.format(i, len(data_loader)))
            x = x.cuda()
            x_var = Variable(x)
            feats = self.feature.forward(x_var)
            
            if all_feats is None:
                all_feats = f.create_dataset('all_feats', [max_count] + list( feats.size()[1:]) , dtype='f')
            all_feats[count:count+feats.size(0)] = feats.data.cpu().numpy()
            all_labels[count:count+feats.size(0)] = y.cpu().numpy()
            count = count + feats.size(0)
        count_var = f.create_dataset('count', (1,), dtype='i')
        count_var[0] = count

        f.close()

    def set_forward_feature(self,x, is_feature = True):
        assert is_feature == True, 'Baseline only support testing with feature'
        
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous().view(self.n_way* self.n_support, -1 )
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        y_support = Variable(y_support.cuda())

        if self.loss_type == 'softmax':
            linear_clf = nn.Linear(self.feat_dim, self.n_way)

        elif self.loss_type == 'dist':        
            linear_clf = backbone.distLinear(self.feat_dim, self.n_way)
        
        linear_clf = linear_clf.cuda()
        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()
        
        batch_size = 4
        support_size = self.n_way* self.n_support

        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()
                z_batch = z_support[selected_id]

                scores = linear_clf(z_batch)
                y_batch = y_support[selected_id] 

                loss = loss_function(scores,y_batch)
                loss.backward()
                set_optimizer.step()

        scores = linear_clf(z_query)
        return scores

    def feature_evaluation(self, cl_data_file, n_way = 5, n_support = 5, n_query = 15, adaptation = False):
        class_list = cl_data_file.keys()

        select_class = random.sample(class_list,n_way)
        z_all  = []
        for cl in select_class:
            img_feat = cl_data_file[cl]

            perm_ids = np.random.permutation(len(img_feat)).tolist()
            z_all.append( [ np.squeeze( img_feat[perm_ids[i]]) for i in range(n_support+n_query) ] )     # stack each batch

        z_all = torch.from_numpy(np.array(z_all) )
        self.n_query = n_query

        scores  = self.set_forward_feature(z_all)
        
        pred = scores.data.cpu().numpy().argmax(axis = 1)
        y = np.repeat(range( n_way ), n_query )
        acc = np.mean(pred == y)*100 
        return acc

    
    def test_loop(self, test_loader, return_std = False): #overwrite parrent function
        correct =0
        count = 0
        acc_all = []
        iter_num = len(test_loader) 
        for i, (x,_) in enumerate(test_loader):
            print i
            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "MAML do not support way change"
            correct_this, count_this = self.correct(x)
            print "test"
            print correct_this/ count_this *100
            print "\n"

            acc_all.append(correct_this/ count_this *100 )

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(test_num,  acc_mean, 1.96* acc_std/np.sqrt(test_num)))
        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean


