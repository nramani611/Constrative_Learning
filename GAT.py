import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch.nn as nn

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()

        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout = 0.6)
        #self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)
        #print(self.conv1.weight.dtype)

    def forward(self, x, edge_index):
        #x = F.dropout(x, p=0.6)
        #print(x)
        #print(x.size())
        print(edge_index.size())
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, p=0.6)
        #x = self.conv2(x, edge_index)
        return x
