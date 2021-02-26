
"""
Create Multiplex Functions.py

@author: taikannakada
"""


import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import random
import os


def load_network(file_name):
    """
    TO-DO:
    rewrite this to accomodate for the working directory of user
    """
    path = '/Users/taikannakada/LINK_PREDICTION_PROJECT/data/' + file_name

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
    fascinate_layers = []

    if fascinate: # for layer Bi
        num_layers = data['F'].shape[1]
        for i in range(num_layers):
            fascinate_layers.append(data['F'][0,i][0][0][0])
        return fascinate_layers

    else: # for layer Ai
        num_layers = data['G'].shape[1]
        for i in range(num_layers):
            layers.append(data['G'][0,i][0][0][0].toarray())
        return layers



def edgelist_A(layer):
    """
    edgelist format for MultiVERSE:
    (layer, source, target, weight)
    """
    edges = np.transpose(np.nonzero(layer))
    ones = np.ones((edges.shape[0],1)).astype(int)
    new_edges = np.hstack((ones, edges, ones))
    return new_edges



def edgelist_B(layer):
    """
    edgelist format for MultiVERSE:
    (layer, source, target, weight)
    """
    mean = np.mean(layer)
    std = np.std(layer)
    threshold = mean + 3*std
    if threshold > 1:
        threshold = .99
    above_threshold = layer > threshold

    edges = np.transpose(np.where(above_threshold)).astype(int)
    ones = np.ones((edges.shape[0],1)).astype(int)
    twos = ones * 2
    new_edges = np.hstack((twos, edges, ones))
    return new_edges



def looped_cosine_similarity(fascinate_layers):
    """
    Cosine_similarity:
    Instead of computing the cos_sim between F and F, we can define a for loop:
    In each loop i, the cos_sim_i is computed beween F and F[i*N:(i+1)*N,], where
    N is the number of rows to be computed in each loop
    """
    N = 500
    cos_sims = []

    for layer in fascinate_layers:
        cos_sim_layer = []
        for i in range(int(layer.shape[0]/N)+1):
            if (i+1)*N > layer.shape[0]:
                final_idx = layer.shape[0]
            else:
                final_idx = (i+1)*N
            cos_sim_i = cosine_similarity(layer, layer[i*N:final_idx])
            cos_sim_layer.append(cos_sim_i)
        cos_sim_layer = np.concatenate(cos_sim_layer, axis=1)
        cos_sims.append(cos_sim_layer)

    return cos_sims



def create_multiplex(file_name):
    """
    create multiplex network for each layer of HMLN
    """
    network = load_network(file_name)

    # for Ai of multiplex network
    layers = layer_as_array(network)

    edges_A = []
    for layer in layers:
        edges_A.append(edgelist_A(layer))

    # for Bi of multiplex network
    fascinate_layers = layer_as_array(network, fascinate=True)
    cos_sims = looped_cosine_similarity(fascinate_layers)

    edges_B = []
    for each in cos_sims:
        edges_B.append(edgelist_B(each))

    # construct multiplex from Ai, Bi
    multiplex_networks = []
    for i in range(len(edges_A)):
        each_multiplex = np.vstack((edges_A[i], edges_B[i]))
        multiplex_networks.append(each_multiplex)

    return multiplex_networks



multiplex = create_multiplex('infra5.mat')

######################################
# Save layers as .txt for MULTIVERSE #
######################################

for i in range(len(multiplex)):
    np.savetxt('infra5_edges_layer' + str(i+1) + '.txt', multiplex[i], fmt='%d', delimiter=' ')
    os.replace('infra5_edges_layer' + str(i+1) + '.txt', './data/infra5_layers/' + 'infra5_edges_layer' + str(i+1) + '.txt')
