import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

class EMT_Dataset(Dataset):
    def __init__(self, scRNA_data, string_2_protein):
        filename = '9606.protein.links.v12.0.txt'

        file = open(filename, 'r')
        lines = file.readlines()
        lines.pop(0)

        string_2_index = dict()
        counter = 0
        for string_id in string_2_protein:
            string_2_index[string_id] = counter
            counter += 1

        list_network = list()
        self.node_features = list()
        self.list_outputs = list()

        print('Getting network tensor...')
        for line in tqdm(lines):
            line = line.strip().split(' ')

            if int(line[2]) >= 999:

                try:
                    id1 = string_2_index[line[0]]
                    id2 = string_2_index[line[1]]
                    list_network.append([id1, id2])
                    list_network.append([id2, id1])

                except KeyError:
                    continue

        print('Getting node features tensor...')
        T0_column_vals = [column for column in scRNA_data.columns if 'T0' in column]
        T8_column_vals = [column for column in scRNA_data.columns if 'T7' in column]
        #print(T8_column_vals)
        proteins = [string_2_protein[string_id] for string_id in string_2_index]

        for column in T0_column_vals:
            self.node_features.append([[scRNA_data.loc[protein, column]] for protein in proteins])
            self.list_outputs.append([1,0])
        #print(self.node_features[0])
        for column in T8_column_vals:
            #print('Hello')
            self.node_features.append([[scRNA_data.loc[protein, column]] for protein in proteins])
            self.list_outputs.append([0,1])

        #print([0, 1] in self.list_outputs)
        #print(len(self.node_features[0]))
        self.edge_index = torch.tensor(list_network).t().contiguous()
        self.node_features = torch.tensor(self.node_features, dtype=torch.float)
        self.list_outputs = torch.tensor(self.list_outputs, dtype=torch.float)

    def __getitem__(self, idx):
        graph = self.edge_index
        node_feature = torch.transpose(self.node_features[idx], 0, 1).t()
        output = self.list_outputs[idx]

        #self.idx += 1
        #if self.idx == len(self.node_features):
            #self.idx = 0

        return node_feature, graph, output

    def __len__(self):
        return len(self.node_features)

    def num_features(self):
        return self.node_features[0].shape[0]

    def num_classes(self):
        return self.list_outputs[0].shape[0]
