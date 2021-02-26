
"""
train.py

@author: taikannakada
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
import random
import create_multiplex_functions as f
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score



############################
# After running MULTIVERSE #
############################

infra_node_embeddings = []

"""
infra_node_embeddings[0]['1'] --> embedding of first node of first layer
"""
for i in range(len(f.multiplex)):
    layer_embed = np.load('/Users/taikannakada/LINK_PREDICTION_PROJECT/data/infra5_embeddings/infra5_edges_layer'+ str(i+1) + '.txtembeddings_M.npy', allow_pickle=True)
    embeddings = layer_embed[()]
    infra_node_embeddings.append(embeddings)


# DO represents 50% of the total cross-layer connections

cross_layer_connections = [[2,3],[2,4],[3,5],[4,5],
                           [1,2],[1,3],[1,4],[1,5]]

DO = []        # connectivity matrix
DO_edges = []  # (layer, source, target, layer)
DO_labels = [] # link labels --> 1: positive, 0: negative

infra5 = f.load_network('infra5.mat')
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



"""
BELOW IS CODE FOR TRAINING

USING 'DO' OF FASCINATE

1. NEGATIVE SAMPLING OF DO --> PRODUCE AS MANY NEGATIVE SAMPLES AS POSITIVE

2. APPLY BINARY OPERATOR ON EMBEDDINGS OF SOURCE AD TARGET NODE OF EACH SAMPLED EDGE
    [BINARY OPERATORS:   - HADAMARD
                         - L1
                         - L2
                         - AVERAGE]

3. TRAIN LOGISTIC REGRESSION CLASSIFIER TO PREDICT BINARY VALUE (EDGE OR NOT)

4. EVALUATE PERFORMANCE
"""

############################################################
# 1. Negative Sampling                                     #
#                                                          #
# Sampling as many negative samples as there are positive  #
#                                                          #
# NODES:                                                   #
# layer1: 39                                               #
# layer2: 151                                              #
# layer3: 61                                               #
# layer4: 47                                               #
# layer5: 51                                               #
#                                                          #
############################################################


for i in range(len(DO_edges)):
    '''
    negative sampling
    '''
    ones = np.ones((DO_edges[i].shape[0],1)).astype(int)
    zeros = np.zeros((DO_edges[i].shape[0],1)).astype(int)
    start = cross_layer_connections[i][0] * ones
    end = cross_layer_connections[i][1] * ones

    a = np.random.randint(DO[i].shape[0], size=(DO_edges[i].shape[0],1))
    b = np.random.randint(DO[i].shape[1], size=(DO_edges[i].shape[0],1))
    sampled_edges = np.hstack((start, a, b, end))

    DO_edges[i] = np.vstack((DO_edges[i], sampled_edges))
    DO_labels[i] = np.vstack((DO_labels[i], zeros))



for i in range(len(DO_edges)):
    if i == 0:
        all_cross_link = DO_edges[i]
        all_cross_link_labels = DO_labels[i]
    else:
        all_cross_link = np.vstack((all_cross_link, DO_edges[i]))
        all_cross_link_labels = np.vstack((all_cross_link_labels, DO_labels[i]))

all_cross_link_labels = all_cross_link_labels.reshape((all_cross_link_labels.shape[0],))


#############################################################################################
# 2. APPLYING BINARY OPERATOR ON EMBEDDINGS OF SOURCE AND TARGET NODES OF EACH SAMPLED EDGE #
#############################################################################################


def operator_hadamard(u,v):
    return u*v

def operator_l1(u,v):
    return np.abs(u-v)

def operator_l2(u,v):
    return (u-v)**2

def operator_avg(u,v):
    return (u+v)/2.0

binary_operators = [operator_hadamard, operator_l1, operator_l2, operator_avg]


# embeddings of source and target nodes of sampled edge
def link_examples_to_features(infra_node_embeddings, all_cross_link, binary_operator):
    '''
    binary operator on source and target nodes

    start_layer = each[0]
    end_layer = each[3]
    start_node = each[1]+1 # 0-index --> node nummber is i+1
    end_node = each[2]+1   # 0-index --> node nummber is i+1
    '''
    return [
        binary_operator(infra_node_embeddings[each[0] - 1][str(each[1]+1)].reshape((128,)), infra_node_embeddings[each[3] - 1][str(each[2]+1)].reshape((128,)))
        for each in all_cross_link
    ]


#################################################################################
# 3. TRAIN LOGISTIC REGRESSION CLASSIFIER TO PREDICT BINARY VALUE (EDGE OR NOT) #
# 4. EVALUATE PERFORMANCE                                                       #
#################################################################################


def link_prediction_classifier(max_iter=10000):
    """
    Logistic Regression:
    Cs:       inverse for regularization strength
    cv:       default cross-validation generator is stratified k-folds. cv is number of folds used
    scoring:  scoring function
    max_iter: max number of iterations of optimization algorithm

    Pipeline:
    steps: list of (name, transform) tuples (implementing fit/transform) that are chained
    """
    lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring='roc_auc', max_iter=max_iter)
    return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])


def train_link_prediction_model(infra_node_embeddings, all_cross_link, link_labels, binary_operator):
    """
    """
    clf = link_prediction_classifier()
    link_features = link_examples_to_features(infra_node_embeddings, all_cross_link, binary_operator)
    clf.fit(link_features, link_labels)
    return clf


def evaluate_link_prediction_model(clf, infra_node_embeddings, all_cross_link, link_labels, binary_operator):

    link_features_test = link_examples_to_features(infra_node_embeddings,
                                                   all_cross_link,
                                                   binary_operator)
    score = evaluate_roc_auc(clf, link_features_test, link_labels)
    return score


def evaluate_roc_auc(clf, link_features, link_labels):
    predicted = clf.predict_proba(link_features)

    # check which class corresponds to positive links
    positive_column = list(clf.classes_).index(1)
    return roc_auc_score(link_labels, predicted[:, positive_column])


def run_link_prediction(binary_operator):
    clf = train_link_prediction_model(infra_node_embeddings,
                                      all_cross_link,
                                      all_cross_link_labels,
                                      binary_operator)

    score = evaluate_link_prediction_model(clf,
                                           infra_node_embeddings,
                                           all_cross_link,
                                           all_cross_link_labels,
                                           binary_operator)

    return {
        "classifer": clf,
        "binary_operator": binary_operator,
        "score": score,
    }



#############
# RUN MODEL #
#############

results = [run_link_prediction(operator) for operator in binary_operators]
best_result = max(results, key=lambda result: result['score'])

print(f"Best result from '{best_result['binary_operator'].__name__}': {best_result['score']}")


# pd.DataFrame(
#     [(result['binary_operator'].__name__, result['score']) for result in results],
#     columns=('name', 'ROC AUC score'),
#     ).set_index('name')
