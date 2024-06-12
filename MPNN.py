import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch import nn
from torch_geometric.utils import dense_to_sparse

#my MPNN method
class MyGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,num_layers, dropout):
        super(MyGCN, self).__init__()
        self.conv = nn.ModuleList()
        self.Linear = nn.ModuleList()
        self.conv.append(GCNConv(in_channels, hidden_channels))
        self.Linear.append(nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                            nn.Dropout(dropout),
                            nn.ReLU()
        ))

        for i in range(num_layers-3):
            self.conv.append(GCNConv(hidden_channels, hidden_channels))
            self.Linear.append(nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.Dropout(dropout),
                nn.ReLU()
            ))
        
        self.conv.append(GCNConv(hidden_channels, hidden_channels))
        self.Linear.append(nn.Linear(hidden_channels, out_channels))


    def forward(self, adjacent, features):
        edge_index,_ = dense_to_sparse(adjacent)
        for i,layer in enumerate(self.conv):
            features = layer(features, edge_index)
            output = self.Linear[i](features)
        return output

