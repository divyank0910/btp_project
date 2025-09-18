import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, to_hetero

class GNNEncoder(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class LinkPredictor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin1 = nn.Linear(2 * in_channels, in_channels)
        self.lin2 = nn.Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        x = torch.cat([z_src, z_dst], dim=-1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return x.squeeze(-1)

class SMR_Model(nn.Module):
    def __init__(self, data, hidden_channels, out_channels):
        super().__init__()
        self.patient_emb = nn.Embedding(data['patient'].num_nodes, hidden_channels)
        self.disease_emb = nn.Embedding(data['disease'].num_nodes, hidden_channels)
        self.medicine_emb = nn.Embedding(data['medicine'].num_nodes, hidden_channels)
        encoder = GNNEncoder(hidden_channels, out_channels)
        self.encoder = to_hetero(encoder, data.metadata(), aggr='sum')
        self.predictor = LinkPredictor(out_channels)

    def forward(self, data, edge_label_index):
        x_dict = {
            "patient": self.patient_emb.weight,
            "disease": self.disease_emb.weight,
            "medicine": self.medicine_emb.weight,
        }
        z_dict = self.encoder(x_dict, data.edge_index_dict)
        src = z_dict['patient'][edge_label_index[0]]  # correct indexing
        dst = z_dict['medicine'][edge_label_index[1]]
        return self.predictor(src, dst)

    def get_all_embeddings(self, data):
        self.eval()
        with torch.no_grad():
            x_dict = {
                "patient": self.patient_emb.weight,
                "disease": self.disease_emb.weight,
                "medicine": self.medicine_emb.weight,
            }
            z_dict = self.encoder(x_dict, data.edge_index_dict)
        return z_dict
