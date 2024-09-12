from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import pool

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()

        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout = 0.6)
        self.output_layer = nn.Linear(out_channels*heads, 2)
        self.conv2 = GATConv(hidden_channels*heads, out_channels, heads, dropout=0.6)
        #print(self.conv1.weight.dtype)

    def forward(self, x, edge_index, tensor_batch):
        #batch_size = x.shape[0]//18840
        #x = F.dropout(x, p=0.6)
        #print(x.size())
        #print(edge_index.size())
        #print(edge_index)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #print(x.size())
        #x = torch.reshape(x, (batch_size, 18840, 8))
        #print(x.size())
        #x = torch.mean(x, dim=2)
        #print(x.size())
        #x = torch.flatten(x)
        x = pool.global_mean_pool(x, tensor_batch)
        x = self.output_layer(x)
        #print(x.size())
        x = F.softmax(x)
        #x = F.dropout(x, p=0.6)
        #x = self.conv2(x, edge_index)
        return x
 
