from torch_geometric.nn import GATConv
import torch.nn as nn
import torch
import torch.nn.functional as F

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        
        self.conv1 = GATConv(in_channels, out_channels, heads, dropout = 0.6)
        self.output_layer = nn.Linear(hidden_channels*out_channels, out_channels)
        #self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)
        #print(self.conv1.weight.dtype)
        
    def forward(self, x, edge_index):
        #x = F.dropout(x, p=0.6)
        #print(x.size())
        #print(edge_index.size())
        #print(edge_index)
        x = self.conv1(x, edge_index)
        #print(x.size())
        x = torch.flatten(x)
        x = self.output_layer(x)
        #print(x.size())
        x = F.softmax(x)
        #x = F.dropout(x, p=0.6)
        #x = self.conv2(x, edge_index)
        return x
