import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.loader import DataLoader
from torch import nn
from utils import Neighbor, commonNeighbor, Union
from MPNN import MyGCN
import networkx as nx
from torch_geometric.utils import to_networkx
import numpy as np
import random

class MLP(nn.Module):
    def __init__(self, In, Hidden, Out, dropout):
        super(MLP,self).__init__()
        self.network = nn.Sequential(
            nn.Linear(In, Hidden),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.LayerNorm(Hidden),
            nn.Linear(Hidden, Hidden),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(Hidden, Hidden),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(Hidden, Out),
            nn.Dropout(dropout)
        )

    def forward(self, data):
        output = self.network(data)
        return output


class NCN(nn.Module):
    def __init__(self, device) -> None:
        super(NCN, self).__init__()
        self.device = device

    def ComputeZij(self, targetLink, adjacent, NodeEmbedding):
        i= targetLink[0]
        j = targetLink[1]
        i_embedding = NodeEmbedding[i]
        j_embedding = NodeEmbedding[j]
        product = i_embedding*j_embedding
        Neighborsrc = Neighbor(adjacent, [i])
        Neighbortar = Neighbor(adjacent, [j])
        #neighborUnion = Union(Neighborsrc, Neighbortar)
        CNlist = commonNeighbor(Neighborsrc, Neighbortar)
        CNproduct = torch.zeros(product.shape[0]).to(self.device)
        #print("CNproduct", CNproduct.shape)

        for idx, CN in enumerate(CNlist):
            for u in CN:
                CNproduct = CNproduct+NodeEmbedding[u]

        product = torch.cat([product, CNproduct], dim = 0)
        return product
     
        
class NCNC2(nn.Module):
    def __init__(self, num_features, dropout, batch_size, device) -> None:
        super(NCNC2, self).__init__()
        self.batch_size = batch_size
        
        self.ncnmlp = MLP(int(0.5*num_features),2*num_features, 1, dropout).to(device)
        self.mlp = MLP(num_features,2*num_features, 1, dropout).to(device)
        self.ncn = NCN(device).to(device)
        for param in self.ncn.parameters():
            param.requires_grad = True
        for param in self.mlp.parameters():
            param.requires_grad = True 
        for param in self.ncnmlp.parameters():
            param.requires_grad = True 
        self.device = device
        
    
    def computP(self, nodeLists, adjacent, NodeEmbedding, source, target):
        
        self.P = [0 for i in range(len(nodeLists))]
        sigmoid = nn.Sigmoid()
        
        
        for i, NeighborOnWalk in enumerate(nodeLists):

            z = torch.prod(NodeEmbedding[NeighborOnWalk], dim = 0)
            z *= NodeEmbedding[source[i]]*NodeEmbedding[target[i]]
            self.P[i] = sigmoid(self.ncnmlp(z))
        
    
    
    def random_walk(self, adjacent, start_node, walk_length, num_walk):
        walk = [[start_node.item()] for idx in range(num_walk)]
        current_node = start_node
        tmp_adj = adjacent.to('cpu')

        for i in range(num_walk):
            for j in range(walk_length - 1):
                neighbors = torch.nonzero(tmp_adj[current_node.item()])
                if neighbors.numel() == 0:
                    walk[i].append(current_node.item())
                    continue
                next_node = random.choice(neighbors)
                walk[i].append(next_node.item())
                current_node = next_node
        
        return torch.tensor(walk)
        

    def forward(self, targetLink, adjacent, NodeEmbedding, G):
        i = targetLink[0]
        j = targetLink[1]

        i_embedding = NodeEmbedding[i]
        j_embedding = NodeEmbedding[j]
        product = i_embedding*j_embedding
        
        #self.computP([i,j], adjacent, NodeEmbedding)

        allpossibleCNproduct = torch.zeros((len(i), product.shape[1])).to(self.device)
        
        for idx in range(len(i)):
            walks_i = self.random_walk(adjacent, i[idx], 3, 10)
            walks_j = self.random_walk(adjacent, j[idx], 3, 10)
            NodesOnWalk = torch.cat([walks_i,walks_j], dim = 0).to(self.device)
            self.computP(NodesOnWalk, adjacent, NodeEmbedding, i, j)
            
            #allpossibleCNproduct[idx] += self.attension(NodeEmbedding, ReferenceLink)
            for k in range(len(NodesOnWalk)):
                z = torch.prod(NodeEmbedding[NodesOnWalk[k]], dim = 0)
                allpossibleCNproduct[idx] = allpossibleCNproduct[idx] + self.P[k]*z

        #print("allpossibleCNproduct shape:", allpossibleCNproduct.shape)
        finalproduct = torch.cat([product, allpossibleCNproduct], dim = 1).to(self.device)
        final_pred = self.mlp(finalproduct)

        return final_pred

