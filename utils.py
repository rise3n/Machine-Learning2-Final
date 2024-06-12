import torch
import math
import random
from torch_geometric.utils import degree, to_dense_adj
from matplotlib import pyplot as plt
import numpy as np
from torch import nn

def Neighbor(adjacent, Nodes):
    
    Neighborlist = []
    for node in Nodes:
        neighbor_indices = torch.nonzero(adjacent[node]).view(-1)
        Neighborlist.append(neighbor_indices)
    
    return Neighborlist


def commonNeighbor(neighborsrc, neighbortar):
    neighborsrc:list[torch.tensor]
    neighbortar:list[torch.tensor]

    list1 = [tensor.tolist() for tensor in neighborsrc]
    list2 = [tensor.tolist() for tensor in neighbortar]
    common_elements = set(map(tuple, list1)) & set(map(tuple, list2))
    common_elements_list = [torch.tensor(element) for element in common_elements]

    return common_elements_list


def Union(neighborsrc, neighbortar, device):
    U = []
    for i,j in zip(neighborsrc, neighbortar):
        allNeighbor = set(i.tolist()).union(set(j.tolist()))
        U.append(torch.tensor(list(allNeighbor)).to(device))
    return U
    
def TargetLinkRemoval(training_edges, targetList, device):
    dense_adj = to_dense_adj(training_edges)[0].to(device)
    for idx in range(targetList.shape[1]):
        i = targetList[0][idx]
        j = targetList[1][idx]
        if dense_adj[i][j] != 1:
            raise Exception("edge not exist")
        else:
            dense_adj[i][j] = 0
    
    #sparse_adj = torch.sparse_coo_tensor(dense_adj.nonzero().T, torch.ones(dense_adj.nonzero().shape[0]).to(device)).to(device)
    #return sparse_adj
    return dense_adj


def split_in_half(dataset, label):
    firsthalf_dataset = dataset[:,:int(0.5*dataset.shape[1])]
    secondhalf_dataset = dataset[:,int(0.5*dataset.shape[1]):]
    firsthalf_label = label[:int(0.5*dataset.shape[1])]
    secondhalf_label = label[int(0.5*dataset.shape[1]):]

    return firsthalf_dataset, firsthalf_label, secondhalf_dataset, secondhalf_label


def predictionwithLogit(logit):
    threshold = 0.5
    sigmoid = nn.Sigmoid()
    p = sigmoid(logit)
    pred = map(int, p>=threshold)
    return torch.tensor(list(pred))


def confusion_matrix(predictons, labels):
    total = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for pred, label in zip(predictions, labels):
        if(pred == 1 and label == 1):
            total += 1
            TP += 1
        elif(pred == 1 and label == 0):
            total += 1
            FP += 1
        elif(pred == 0 and label == 0):
            total += 1
            TN += 1
        elif(pred == 0 and label == 1):
            total += 1
            FN += 1
        else:
            raise Exception("prediction is not 0 or 1")

    accuracy = (TP+TN)/total
    precision = TP / (TP+FP)
    recall = TP/(TP+FN)
    specificity = TN/(TN+FP)
    F1 = 2*(precision*recall)/(precision+recall)

    return accuracy, precision, recall, specificity, F1


class MyPreTransform(object):
    def __init__(self, ratio):
        self.ratio = ratio
        
    def __call__(self, data):
        edge_index = data.edge_index
        tmp, removed = self.remove_edges(edge_index, self.ratio)            
        data.edge_index = tmp
        self.removed = removed
        return data

    def remove_edges(self, edge_index, ratio):
        num_edges = edge_index.size(1)
        num_edges_to_remove = math.floor(num_edges*(1-ratio))
        indices_to_remove = random.sample(range(num_edges), num_edges_to_remove)
        mask = torch.ones(num_edges, dtype=torch.bool)
        mask[indices_to_remove] = False
        new_edge = edge_index[:, mask]
        removed_edge = edge_index[:,mask== False]
        
        return new_edge, removed_edge


def draw_hist(dataset,title):
    neighbor_counts = []
    for data in dataset:
        edge_index = data.edge_index
        num_neighbors = degree(edge_index[0], num_nodes=data.num_nodes)
        neighbor_counts.extend(num_neighbors.tolist())

    neighbor_count_distribution = torch.histc(torch.tensor(neighbor_counts), bins=int(max(neighbor_counts)))


    plt.bar(range(len(neighbor_count_distribution)), neighbor_count_distribution)
    for i, freq in enumerate(neighbor_count_distribution):
        plt.text(i, freq, str(int(freq)), ha='center', va='bottom')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()