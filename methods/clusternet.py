import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

import copy

class DBLoss(torch.nn.Module):
    def __init__(self):
        super(DBLoss, self).__init__()
        
    def forward(self, db1, db2):
        #return db1 - db2
        return db1 / db2

def DBindex(z_support_flat, z_proto, d1, d2, y_support, n_way, n_support, support_size):

    s =  torch.zeros((n_way, )).cuda()
    for c in range(n_way):
        for ii in range(support_size):
            if y_support[ii] == c:
                s[c] = s[c] + d1(z_support_flat[ii], z_proto[c])
    s = s / n_support

    M = torch.zeros((n_way, n_way)).cuda()
    for c1 in range(n_way):                  
        for c2 in range(n_way):
            M[c1][c2] =  d2(z_proto[c1], z_proto[c2])
    R =  torch.zeros((n_way, n_way)).cuda()

    for c1 in range(n_way):                  
        for c2 in range(n_way):
            if c1 == c2:
                R[c1][c2] = 0.0
            else:
                R[c1][c2] =   (s[c1] + s[c2]) /  M[c1][c2]

    v, _ =  torch.max(R, 1)
    #v =  torch.mean(R, 1)
    db = torch.mean(v)
    return db

def DBindex_test(z_support_flat, z_proto, z_query, j, d1, d2, y_support, n_way, n_support, support_size):

    s =  torch.zeros((n_way, )).cuda()
    for c in range(n_way):

        for ii in range(support_size):
            if y_support[ii] == c:
                s[c] = s[c] + d1(z_support_flat[ii], z_proto[c])
        
        if c == j:
            s[c] = s[c] + d1(z_query, z_proto[c]) 
    
    for c in range(n_way):
        if c != j:
            s[c] = s[c] / n_support    
        else:
            s[c] = s[c] / ( n_support + 1)

    M = torch.zeros((n_way, n_way)).cuda()
    for c1 in range(n_way):                  
        for c2 in range(n_way):
            M[c1][c2] =  d2(z_proto[c1], z_proto[c2])
    R =  torch.zeros((n_way, n_way)).cuda()

    for c1 in range(n_way):                  
        for c2 in range(n_way):
            if c1 == c2:
                R[c1][c2] = 0.0
            else:
                R[c1][c2] =   (s[c1] + s[c2]) /  M[c1][c2]

    v, _ =  torch.max(R, 1)
    #v =  torch.mean(R, 1)
    db = torch.mean(v)
    return db

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)
            )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)
            )

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1))

    def forward(self, input1, input2):
        output1 = self.fc1(input1)
        output2 = self.fc2(input2)
        output = torch.cat((output1, output2), 0)
        output = self.fc(output)
        output = F.sigmoid(output)
        return output

class clusternet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, loss_type = "softmax"):
        super(clusternet, self).__init__( model_func,  n_way, n_support)
        self.loss_type = loss_type

        self.siameseNetwork_class = SiameseNetwork()
        self.siameseNetwork_between = SiameseNetwork()

    def set_forward(self, x, is_feature = True):
        return self.set_forward_adaptation(x, is_feature); #Baseline always do adaptation
 
    def set_forward_adaptation(self, x,  is_feature = True):
        assert is_feature == True, 'Baseline only support testing with feature'
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support_flat   = z_support.contiguous().view(self.n_way* self.n_support, -1 ) # 25 * 512
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 ) # 16 * 512

        '''
        z_support_norm = torch.norm(z_support_flat, p=2, dim =1).unsqueeze(1).expand_as(z_support_flat)
        z_support_flat = z_support_flat.div(z_support_norm+ 0.00001)

        z_query_norm = torch.norm(z_query, p=2, dim =1).unsqueeze(1).expand_as(z_query)
        z_query = z_query.div(z_query_norm+ 0.00001)
        '''

        z_support   = z_support_flat.view(self.n_way, self.n_support, -1 )
        z_proto     = z_support.mean(1) #the shape of z is [n_data, n_dim] # 5 * 512

        y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        y_support = Variable(y_support.cuda())

        
        set_optimizer1 = torch.optim.SGD(self.siameseNetwork_class.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
        set_optimizer2 = torch.optim.SGD(self.siameseNetwork_between.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        ########################################################################
        loss_function = DBLoss()
        loss_function = loss_function.cuda()
        support_size = self.n_way * self.n_support 

        for epoch in range(2):
            for i in range(support_size):
                for j in range(self.n_way):
                    if y_support[i] != j:

                        y_support_j = Variable(y_support.data.clone(), requires_grad=False)
                        z_proto_j = Variable(z_proto.data.clone(), requires_grad=False)
                        
                        y_support_j[i] = j

                        z_proto_j[y_support[i]] = (z_proto_j[y_support[i]] - z_support_flat[i] * (1/self.n_support)) * (self.n_support / (self.n_support-1))
                        z_proto_j[j] = (z_proto_j[j] * (self.n_support / (self.n_support+1)) + z_support_flat[i] * (1/(self.n_support+1)))

                        db1 = DBindex(z_support_flat, z_proto, self.siameseNetwork_class, self.siameseNetwork_between, y_support, self.n_way, self.n_support ,support_size)
                        db2 = DBindex(z_support_flat, z_proto_j, self.siameseNetwork_class, self.siameseNetwork_between, y_support, self.n_way, self.n_support, support_size)
                        print db1
                        print db2
                        print "\n"

                        loss = loss_function(db1, db2)
                        set_optimizer1.zero_grad()
                        set_optimizer2.zero_grad()

                        loss.backward()
                        set_optimizer1.step()
                        set_optimizer2.step()
        
        
        pred = np.zeros((z_query.size()[0], ))

        for i in range(z_query.size()[0]):
            s =  torch.zeros((self.n_way, )).cuda()

            for j in range(self.n_way):

                z_proto_j = Variable(z_proto.data.clone(), requires_grad=False)            
                z_proto_j[j] = (z_proto_j[j] * (self.n_support / (self.n_support+1)) + z_query[i] * (1/(self.n_support+1)))
                db = DBindex_test(z_support_flat, z_proto_j, z_query[i], j, self.siameseNetwork_class, self.siameseNetwork_between, y_support, self.n_way, self.n_support, support_size)
            
                s[j] = db

            print s
            pred[i] = s.data.cpu().numpy().argmin()
            print pred
        #############################################################################
        '''
        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()
        batch_size = 4
        support_size = self.n_way * self.n_support

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
        '''
        ###########################################################################
        return pred

    def set_forward_loss(self,x):
        raise ValueError('Baseline predict on pretrained feature and do not support finetune backbone')
