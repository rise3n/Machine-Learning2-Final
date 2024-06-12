import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.loader import DataLoader
from torch import nn
from utils import Neighbor, commonNeighbor, Union
from MPNN import MyGCN


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
     
        
class NCNC(nn.Module):
    def __init__(self, num_features, dropout, batch_size, device) -> None:
        super(NCNC, self).__init__()
        self.batch_size = batch_size
        
        self.ncnmlp = MLP(num_features,2*num_features, 1, dropout).to(device)
        self.mlp = MLP(num_features,2*num_features, 1, dropout).to(device)
        self.ncn = NCN(device).to(device)
        #self.attention = GATConv(num_features, num_features, heads = num_head, concat = False).to(device)
        for param in self.ncn.parameters():
            param.requires_grad = True
        for param in self.mlp.parameters():
            param.requires_grad = True 
        self.device = device
        

    def computP(self, targetLink, adjacent, NodeEmbedding):
        srcNode = targetLink[0] #should be a list
        tarNode = targetLink[1]
        Neighborsrc = Neighbor(adjacent, srcNode)
        Neighbortar = Neighbor(adjacent, tarNode)
        neighborUnion = Union(Neighborsrc, Neighbortar, self.device)
        #CNlist = commonNeighbor(Neighborsrc, Neighbortar)
        self.P = [[] for i in range(len(neighborUnion))]
        sigmoid = nn.Sigmoid()
        
        for i, allNeighbor in enumerate(neighborUnion):
            #print("all neighbor", allNeighbor)
            for j,U in enumerate(allNeighbor):
                u = U.item()
                if u in Neighborsrc[i] and u in Neighbortar[i]:
                    self.P[i].append(1)
                elif u in Neighborsrc[i] and u not in Neighbortar[i]:
                    zij = self.ncn.ComputeZij([U,tarNode[i]], adjacent, NodeEmbedding)
                    A_ju = sigmoid(self.ncnmlp(zij))
                    self.P[i].append(A_ju)
                elif u  not in Neighborsrc[i] and u in Neighbortar[i]:
                    zij = self.ncn.ComputeZij([srcNode[i], U], adjacent, NodeEmbedding)
                    A_iu = sigmoid(self.ncnmlp(zij))
                    self.P[i].append(A_iu)
                elif u not in Neighborsrc[i] and u not in Neighbortar[i]:
                    self.P[i].append(0)


    def forward(self, targetLink, adjacent, NodeEmbedding):
        i = targetLink[0]
        j = targetLink[1]

        
        i_embedding = NodeEmbedding[i]
        j_embedding = NodeEmbedding[j]
        product = i_embedding*j_embedding

        Neighborsrc = Neighbor(adjacent, i)
        Neighbortar = Neighbor(adjacent, j)
        neighborUnion = Union(Neighborsrc, Neighbortar, self.device)
        self.computP([i,j], adjacent, NodeEmbedding)

        allpossibleCNproduct = torch.zeros((len(neighborUnion), product.shape[1])).to(self.device)
        #print(len(neighborUnion))
        for idx in range(len(neighborUnion)):
            for j in range(neighborUnion[idx].shape[0]):

                allpossibleCNproduct[idx] = allpossibleCNproduct[idx]+self.P[idx][j]*NodeEmbedding[neighborUnion[idx][j]]

        #print("allpossibleCNproduct shape:", allpossibleCNproduct.shape)
        finalproduct = torch.cat([product, allpossibleCNproduct], dim = 1)
        final_pred = self.mlp(finalproduct)

        return final_pred

