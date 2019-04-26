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

from datasets import svhn_few_shot, cifar_few_shot, caltech256_few_shot


def loss_fn_kd(outputs, labels, teacher_outputs, alpha=0.3):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    T = 1.0
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)
    return KD_loss

class DomainLoss(torch.nn.Module):
    def __init__(self):
        super(DomainLoss, self).__init__()
        
    def forward(self, score1, score2):
        return  torch.mean(- torch.log(F.sigmoid(score1)) - torch.log((1 - F.sigmoid(score2))))

def test_loop(novel_loader, base_loader, return_std = False, loss_type="softmax"): #overwrite parrent function
    correct = 0
    count = 0
    teacher_acc_all = []


    iter_num = len(novel_loader) 
    for i, (x,_) in enumerate(novel_loader):
        print i
        print x.size()
        start = time.time()
        ###############################################################################################
        if params.dataset == "omniglot_to_emnist":
            student           = knowledge(  backbone.Conv4S_noflatten)
        else:
            student           = knowledge(  backbone.ResNet18_noflatten)
 
        method = "baseline"
        ###############################################################################################      
        checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, method)
        if params.train_aug:
            checkpoint_dir += '_aug'
        print checkpoint_dir

        params.save_iter = 399
        if params.save_iter != -1:
            modelfile   = get_assigned_file(checkpoint_dir, params.save_iter)
        elif params.method in ['baseline', 'baseline++'] :
            modelfile   = get_resume_file(checkpoint_dir)
        else:
            modelfile   = get_best_file(checkpoint_dir)

        tmp = torch.load(modelfile)
        state = tmp['state']
        state = {k: v for k, v in state.items() if 'classifier' not in k}
        student.load_state_dict(state)

        if params.dataset == "omniglot_to_emnist":
            partial = nn.Sequential(*[student.feature.trunk[i] for i in range(3)])
        else:
            partial = nn.Sequential(*[student.feature.trunk[i] for i in range(11)])

        ###############################################################################################
        student.n_query = x.size(1) - student.n_support
        x = x.cuda()
        x_var = Variable(x)

        x_a_i = x_var[:,:student.n_support,:,:,:].contiguous().view( student.n_way* student.n_support, *x.size()[2:]) 
        x_b_i = x_var[:, student.n_support:,:,:,:].contiguous().view( student.n_way* student.n_query,   *x.size()[2:]) 

        y_a_i = Variable( torch.from_numpy( np.repeat(range( student.n_way ), student.n_support ) )).cuda()
        batch_size = 4
        support_size = student.n_way * student.n_support

        if loss_type == 'softmax':

            if params.dataset == "omniglot_to_emnist":
                feature_transformation =  nn.Sequential(
                    student.feature.trunk[3],
                    backbone.Flatten()).cuda()
            else:
                feature_transformation =  nn.Sequential(
                    student.feature.trunk[11],
                    nn.AvgPool2d(7),
                    backbone.Flatten()).cuda()

            feature = nn.Sequential(
                partial,
            ).cuda()
            
            feature_flatten = nn.Sequential(
                nn.AvgPool2d(7),
                backbone.Flatten())       

            model_classifier = nn.Linear(512, student.n_way).cuda()
            model_classifier.bias.data.fill_(0)

            domain_discriminator1 = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            ) 
            domain_discriminator1.cuda()

            domain_discriminator2 = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            ) 
            domain_discriminator2.cuda()

        elif loss_type == 'dist': #Baseline ++
            student.classifier = backbone.distLinear(student.feature.final_feat_dim, student.n_way).cuda()
            teacher.classifier = backbone.distLinear(student.feature.final_feat_dim, student.n_way).cuda()
        
        loss_fn = nn.CrossEntropyLoss().cuda()
        loss_domain = DomainLoss().cuda()
        loss_variation = nn.MSELoss().cuda()

        classifier_opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model_classifier.parameters()), lr = 0.01, momentum=0.9)
        feature_transformation_opt = torch.optim.SGD(filter(lambda p: p.requires_grad, feature_transformation.parameters()), lr = 0.01, momentum=0.9)

        feature_opt = torch.optim.SGD(filter(lambda p: p.requires_grad, feature.parameters()), lr = 0.01, momentum=0.9)
        domain_opt1 = torch.optim.SGD(domain_discriminator1.parameters(), lr=0.01)
        domain_opt2 = torch.optim.SGD(domain_discriminator2.parameters(), lr=0.01)

        ###############################################################################################

        feature.train()
        feature_transformation.train()
        model_classifier.train()

        domain_discriminator1.train()
        domain_discriminator2.train()

        for i, (x_,y_) in enumerate(base_loader):
            x = x_
            y = y_  
            break

        for epoch in range(100):
            x = Variable(x.cuda())
            y = Variable(y.cuda())

            loss_ = 0
            rand_id = np.random.permutation(support_size)

            for i in range(0, support_size , batch_size):

                domain_opt1.zero_grad()
                domain_opt2.zero_grad()
                feature_opt.zero_grad()
                classifier_opt.zero_grad()
                feature_transformation_opt.zero_grad()

                #####################################
                selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size)]).cuda()
                z_batch = x_a_i[selected_id]
                source_batch = x[selected_id]

                y_batch = y_a_i[selected_id] 

                #####################################
                
                target_feature = feature(z_batch)
                target_feature_transformed = feature_transformation(target_feature)
                outputs = model_classifier(target_feature_transformed)
                
                #####################################
                
                source_feature = feature(source_batch) 
                #####################################
                source_feature = feature_flatten(source_feature)
                target_feature = feature_flatten(target_feature)

                source_score = domain_discriminator1(source_feature)
                target_score = domain_discriminator1(target_feature)
                ###################################################################
                
                target_score2 = domain_discriminator2(target_feature_transformed)
                source_score2 = domain_discriminator2(source_feature)

                ###################################################################
                loss_domain_classifier1 =  0.1*loss_domain(source_score, target_score)

                loss_domain_classifier1.backward(retain_graph=True)
                domain_opt1.step()

                loss_domain_classifier2 =  0.1*loss_domain(target_score2, source_score2)
                loss_domain_classifier2.backward(retain_graph=True)
                domain_opt2.step()

                domain_feature1 = 0.1*loss_domain(target_score, source_score)
                domain_feature2 = 0.1*loss_domain(source_score2, target_score2)
            
                loss = loss_fn(outputs, y_batch)
                loss = loss + domain_feature1

                loss.backward(retain_graph=True)
                feature_opt.step()
                classifier_opt.step()

                domain_feature2.backward()
                feature_transformation_opt.step()

            outputs = feature(x_a_i)
            outputs = feature_transformation(outputs)
            z_support  = outputs.view(student.n_way, student.n_support, -1 ) # 5 * 5 * 512
            z_proto    = z_support.mean(1).repeat(1, student.n_support).view(-1, 512) # 5 * 512
            feature_opt.zero_grad()
            loss = student.n_way * student.n_support * loss_variation(outputs, z_proto)
            loss.backward()
            feature_opt.step()
        

        outputs = feature(x_a_i)
        outputs = feature_transformation(outputs)
        scores = model_classifier(outputs)

        y_query = np.repeat(range( student.n_way ), student.n_support )
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        correct_this, count_this = float(top1_correct), len(y_query)
        print "train"
        print correct_this/ count_this *100

        feature.eval()
        feature_transformation.eval()
        model_classifier.eval()

        outputs = feature(x_b_i)
        outputs = feature_transformation(outputs)
        scores = model_classifier(outputs)

        y_query = np.repeat(range( student.n_way ), student.n_query )
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()

        top1_correct = np.sum(topk_ind[:,0] == y_query)
        correct_this, count_this = float(top1_correct), len(y_query)
        print "test"
        print correct_this/ count_this *100
        teacher_acc_all.append(correct_this / count_this *100 )
        alpha = correct_this/ count_this
        
        print "time"
        print time.time() - start
        print "\n"
        ###############################################################################################
    
    acc_all  = np.asarray(teacher_acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')

    if params.dataset == 'miniImageNet_to_CUB':
        base_file = configs.data_dir['miniImagenet'] + 'all.json' 
        val_file   = configs.data_dir['CUB'] + 'val.json' 
    
    elif params.dataset == 'CUB_to_miniImageNet':
        base_file = configs.data_dir['CUB'] + 'base.json' 
        val_file   = configs.data_dir['miniImagenet'] + 'val.json'
        
    elif params.dataset == 'omniglot_to_emnist':
        base_file = configs.data_dir['omniglot'] + 'noLatin.json' 
        val_file   = configs.data_dir['emnist'] + 'val.json' 


    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'cross_char']:
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
        
        elif params.dataset == "cifar100_to_caltech256":
            base_datamgr    = cifar_few_shot.SimpleDataManager(image_size, batch_size = 16)
            base_loader    = base_datamgr.get_data_loader( "base" , aug = True )

            val_datamgr     = caltech256_few_shot.SimpleDataManager(image_size, batch_size = 64)
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
        elif params.method == 'baseline++':
            model           = BaselineTrain( model_dict[params.model], params.num_classes, loss_type = 'dist')

    elif params.method in ['protonet','matchingnet','relationnet', 'relationnet_softmax', 'maml', 'maml_approx', 'featurenet', 'knowledge']:
        n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small

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
       
        elif params.method in ['knowledge']:

            iter_num = 600
            few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) 
            datamgr         = SetDataManager(image_size, n_eposide = iter_num, n_query = 15 , **few_shot_params)
           
            split = "novel"

            if params.dataset == 'miniImageNet_to_CUB':
                base_file = configs.data_dir['miniImagenet'] + 'all.json' 
                novel_file   = configs.data_dir['CUB'] + split +'.json' 
            elif params.dataset == 'CUB_to_miniImageNet':
                base_file = configs.data_dir['CUB'] + 'base.json' 
                novel_file   = configs.data_dir['miniImagenet'] + split +'.json' 
            elif params.dataset == 'omniglot_to_emnist':
                base_file = configs.data_dir['omniglot'] + 'noLatin.json' 
                novel_file   = configs.data_dir['emnist'] + split +'.json' 

            if params.dataset not in ["cifar100_to_caltech256", "caltech256_to_cifar100"]:

                base_datamgr    = SimpleDataManager(image_size, batch_size = 25)
                base_loader     = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
                novel_loader    = datamgr.get_data_loader( novel_file, aug = False)

            elif params.dataset == "cifar100_to_caltech256":
                base_datamgr        = caltech256_few_shot.SetDataManager('novel', image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
                novel_loader        = base_datamgr.get_data_loader(aug =False)
              
                base_datamgr    = cifar_few_shot.SimpleDataManager(image_size, batch_size = 25)
                base_loader    = base_datamgr.get_data_loader( "base" , aug = True )


            elif params.dataset == "caltech256_to_cifar100":
                base_datamgr       = cifar_few_shot.SetDataManager('novel', image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
                novel_loader       = base_datamgr.get_data_loader(aug =False)

                base_datamgr    = caltech256_few_shot.SimpleDataManager(image_size, batch_size = 25)
                base_loader     = base_datamgr.get_data_loader( "base" , aug = True )
    else:
       raise ValueError('Unknown method')

    #########################################################################

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'

    if not params.method  in ['baseline', 'baseline++']: 
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    if params.method == 'maml' or params.method == 'maml_approx' :
        stop_epoch = params.stop_epoch * model.n_task #maml use multiple tasks in one update 

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            model.load_state_dict(tmp['state'])

    elif params.warmup: #We also support warmup from pretrained baseline feature, but we never used in our paper
        baseline_checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, 'baseline')
        if params.train_aug:
            baseline_checkpoint_dir += '_aug'
        warmup_resume_file = get_resume_file(baseline_checkpoint_dir)
        tmp = torch.load(warmup_resume_file)
        if tmp is not None: 
            state = tmp['state']
            state_keys = list(state.keys())
            for i, key in enumerate(state_keys):
                if "feature." in key:
                    newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                    state[newkey] = state.pop(key)
                else:
                    state.pop(key)
            model.feature.load_state_dict(state)
        else:
            raise ValueError('No warm_up file')

    test_loop(novel_loader, base_loader, return_std = False)
