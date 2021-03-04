
"""
Restriced Boltzmann Machine for Link Prediction of Heterogeneous Multi-layered networks

Inspired by "Restricted Boltzmann Machines for Collaborative Filtering" by R. Salakhutdinov,  A. Mnih, G. Hinton

The idea is to treat cross-layer dependence inference problem as a collaboratve filtering problem.

1. nodes from given laayer are objects from given domain (user/items)
2. within-layer connectivity is an object-object similarity measure (how similar are two objects?)
3. cross-layer dependencies are the "ratings" from objects of one domain to those of another domain

Therefore, cross-layer link prediction can be framed as an attempt to inferr the missing ratings between users and items
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from scipy.io import loadmat



class RBM(nn.Module):
    def __init__(self,
               n_vis=151,
               n_hin=39,
               k=5):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hin,n_vis)*1e-2)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hin))
        self.k = k

    def sample_from_p(self,p):
        return F.relu(torch.sign(p - Variable(torch.rand(p.size()))))

    def v_to_h(self,v):
        p_h = torch.sigmoid(F.linear(v,self.W,self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h,sample_h

    def h_to_v(self,h):
        p_v = torch.sigmoid(F.linear(h,self.W.t(),self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v,sample_v

    def forward(self,v):
        pre_h1,h1 = self.v_to_h(v)

        h_ = h1
        for _ in range(self.k):
            pre_v_,v_ = self.h_to_v(h_)
            pre_h_,h_ = self.v_to_h(v_)

        return v,v_

    def free_energy(self,v):
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v,self.W,self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()



def load_network(file_name):
    """
    TO-DO:
    rewrite this to accomodate for the working directory of user
    """
    path = '/Users/taikannakada/FASCINATE/'+ file_name
    data = loadmat(path)
    return data


def layer_as_array(data, fascinate=False):
    '''
    once matlab data is loaded via scipy.io.loadmat
    convert layer data into matrix
    data: loaded matlab data
    fascinate: if True then edge_list for cos_sim matrix
    '''
    layers = []

    num_layers = data['G'].shape[1]
    for i in range(num_layers):
        layers.append(data['G'][0,i][0][0][0].toarray())
    return layers

def edgelist(layer):
    """
    (layer, source, target, weight)
    """
    edges = np.transpose(np.nonzero(layer))
    ones = np.ones((edges.shape[0],1)).astype(int)
    new_edges = np.hstack((ones, edges, ones))
    return new_edges


#################
# PREPARE INPUT #
#################

original_data = load_network('infra5.mat')
data = layer_as_array(original_data)


cross_layer_connections = [[2,3],[2,4],[3,5],[4,5],
                           [1,2],[1,3],[1,4],[1,5]]

DO = []        # connectivity matrix
DO_edges = []  # (layer, source, target, layer)
DO_labels = [] # link labels --> 1: positive, 0: negative

infra5 = load_network('infra5.mat')
num_layers = infra5['DO'].shape[1]

for i in range(num_layers):
    DO.append(infra5['DO'][0,i][0][0][0].toarray())

    edges = np.transpose(np.nonzero(DO[i]))
    ones = np.ones((edges.shape[0],1)).astype(int)

    start = cross_layer_connections[i][0] * ones
    end = cross_layer_connections[i][1] * ones
    new_edges = np.hstack((start, edges, end))

    DO_edges.append(new_edges)
    DO_labels.append(ones)


DO_TEST= []
DO_edges_TEST = []
DO_labels_TEST = [] # link labels --> 1: positive, 0: negative

for i in range(num_layers):
    difference = infra5['DU'][0,i][0][0][0].toarray() - infra5['DO'][0,i][0][0][0].toarray()
    DO_TEST.append(difference)

    edges = np.transpose(np.nonzero(DO_TEST[i]))
    ones = np.ones((edges.shape[0],1)).astype(int)

    start = cross_layer_connections[i][0] * ones
    end = cross_layer_connections[i][1] * ones
    new_edges = np.hstack((start, edges, end))

    DO_edges_TEST.append(new_edges)
    DO_labels_TEST.append(ones)


training_set = torch.FloatTensor(DO[4])
testing_set = torch.FloatTensor(DO_TEST[4])

################
# TRAINING RBM #
################

train_loader = torch.utils.data.DataLoader(training_set)
rbm = RBM(k=1)
train_op = optim.SGD(rbm.parameters(), 0.1)

for epoch in range(10):
    loss_ = []
    for (node, target) in enumerate(train_loader):
        data = Variable(target)
        sample_data = data.bernoulli()

        v, v1 = rbm(sample_data)
        loss = rbm.free_energy(v) - rbm.free_energy(v1)
        loss_.append(loss.data)
        train_op.zero_grad()
        loss.backward()
        train_op.step()

    print("Training loss for {} epoch: {}".format(epoch, np.mean(loss_)))
