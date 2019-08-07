import numpy as np
import torch
from torch.autograd import Variable
import os
import glob
import h5py

import configs
import backbone
from data.datamgr import SimpleDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file 

from datasets import svhn_few_shot, cifar_few_shot,  caltech256_few_shot, ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot


from util import load_pretrained_model


def save_features(model, data_loader, outfile):
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
        feats = model(x_var)

        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list( feats.size()[1:]) , dtype='f')
        all_feats[count:count+feats.size(0)] = feats.data.cpu().numpy()
        all_labels[count:count+feats.size(0)] = y.cpu().numpy()
        count = count + feats.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()

if __name__ == '__main__':
    params = parse_args('save_features')
    assert params.method != 'maml' and params.method != 'maml_approx', 'maml do not support save_feature and run'

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

    split = params.split

    if params.dataset == "CUB_to_miniImageNet":
        loadfile   = configs.data_dir['miniImagenet'] + split +'.json' 
    elif params.dataset == "miniImageNet_to_CUB":
        loadfile   = configs.data_dir['CUB'] + split +'.json' 


    elif params.dataset == "omniglot_to_emnist":
        loadfile  = configs.data_dir['emnist'] + split +'.json' 

    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        checkpoint_dir += '_aug'

    if not params.method in ['baseline', 'baseline++'] :
        checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    if params.save_iter != -1:
        modelfile   = get_assigned_file(checkpoint_dir, params.save_iter)
    elif params.method in ['baseline', 'baseline++'] :
        modelfile   = get_resume_file(checkpoint_dir)
    else:
        modelfile   = get_best_file(checkpoint_dir)

    if params.save_iter != -1:
        outfile = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + "_" + str(params.save_iter)+ ".hdf5") 
    else:
        outfile = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + ".hdf5") 


    if params.dataset not in ["cifar100_to_cifar10", "caltech256_to_cifar100"]:

        #datamgr         = SimpleDataManager(image_size, batch_size = 64)
        #data_loader     = datamgr.get_data_loader(loadfile, aug = False)

        datamgr         = CropDisease_few_shot.SimpleDataManager(image_size, batch_size = 64)
        data_loader     = datamgr.get_data_loader(aug = False )


    elif params.dataset == "cifar100_to_caltech256":
        datamgr         = caltech256_few_shot.SimpleDataManager(image_size, batch_size = 64)
        data_loader     = datamgr.get_data_loader( "novel" , aug = False )

    elif params.dataset == "cifar100_to_cifar10":
        datamgr         = cifar_few_shot.SimpleDataManager("CIFAR10", 224, batch_size = 64)
        data_loader     = datamgr.get_data_loader( "novel" , aug = False )   


    if params.method in ['relationnet', 'relationnet_softmax']:
        if params.model == 'Conv4': 
            model = backbone.Conv4NP()
        elif params.model == 'Conv6': 
            model = backbone.Conv6NP()
        elif params.model == 'Conv4S': 
            model = backbone.Conv4SNP()
        else:
            model = model_dict[params.model]( flatten = False )

    elif params.method in ['maml' , 'maml_approx']: 
       raise ValueError('MAML do not support save feature')
    else:
        model = model_dict[params.model]()
    

    model = model.cuda()
    '''
    branch = 1
    num_classes = 5
    resume = "/home/ibm2019/branchnet/checkpoints/train_from_scratch_flowers/model_best.pth.tar"
    #model = load_pretrained_model.load_multi_branch_pretrained_model(resume, 50, num_classes, False)
    model = load_pretrained_model.load_trained_model(50, num_classes, False, resume=resume)

    model = model.cuda()

    print(model)
    '''

    tmp = torch.load(modelfile)
    state = tmp['state']


    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if "feature." in key:
            newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
            state[newkey] = state.pop(key)
        else:
            state.pop(key)

            
    model.load_state_dict(state)

    
    model.eval()

    dirname = os.path.dirname(outfile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    save_features(model, data_loader, outfile)
