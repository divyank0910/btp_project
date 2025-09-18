import argparse
import torch
import torch.nn as nn
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GATConv, to_hetero
from sklearn.metrics import roc_auc_score, jaccard_score

# Import your data loading function
from data_preprocessing import load_and_process_data

# ---------------- Model Definition ----------------
class GNNEncoder(nn.Module):
    """Graph Neural Network Encoder using GATConv."""
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class GNNModel(nn.Module):
    """Heterogeneous GNN for link prediction with learnable embeddings."""
    def __init__(self, hidden_channels, out_channels, metadata, data):
        super().__init__()
        # Learnable embeddings for each node type
        self.patient_emb = nn.Embedding(data['patient'].num_nodes, hidden_channels)
        self.disease_emb = nn.Embedding(data['disease'].num_nodes, hidden_channels)
        self.medicine_emb = nn.Embedding(data['medicine'].num_nodes, hidden_channels)

        # Encoder
        encoder = GNNEncoder(hidden_channels, out_channels)
        self.encoder = to_hetero(encoder, metadata=metadata, aggr='sum')

        # Predictor
        self.lin = nn.Linear(out_channels * 2, 1)

    def forward(self, data, edge_label_index):
        x_dict = {
            "patient": self.patient_emb.weight,
            "disease": self.disease_emb.weight,
            "medicine": self.medicine_emb.weight,
        }
        z_dict = self.encoder(x_dict, data.edge_index_dict)
        row, col = edge_label_index
        edge_emb = torch.cat([z_dict['patient'][row], z_dict['medicine'][col]], dim=-1)
        return self.lin(edge_emb).view(-1)

# ---------------- BPR Loss ----------------
def bpr_loss(model, data, edge_label_index, edge_label):
    pos_mask = edge_label == 1
    neg_mask = edge_label == 0

    pos_src, pos_dst = edge_label_index[:, pos_mask]
    neg_src, neg_dst = edge_label_index[:, neg_mask]

    if pos_src.numel() == 0 or neg_src.numel() == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    # Balance positive and negative samples
    num_samples = min(pos_src.size(0), neg_src.size(0))
    perm_pos = torch.randperm(pos_src.size(0))[:num_samples]
    perm_neg = torch.randperm(neg_src.size(0))[:num_samples]

    pos_scores = model(data, torch.stack([pos_src[perm_pos], pos_dst[perm_pos]]))
    neg_scores = model(data, torch.stack([neg_src[perm_neg], neg_dst[perm_neg]]))

    return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

# ---------------- Training & Evaluation ----------------
def train(model, train_data, optimizer, loss_fn, alpha=0.5):
    model.train()
    optimizer.zero_grad()

    edge_label_index = train_data['patient', 'takes', 'medicine'].edge_label_index
    edge_label = train_data['patient', 'takes', 'medicine'].edge_label

    # BCE Loss
    pred = model(train_data, edge_label_index)
    bce = loss_fn(pred, edge_label.float())

    # BPR Loss
    bpr = bpr_loss(model, train_data, edge_label_index, edge_label)

    # Combined Loss
    loss = (1 - alpha) * bce + alpha * bpr
    loss.backward()
    optimizer.step()
    return float(loss), float(bce), float(bpr)

@torch.no_grad()
def evaluate(model, data, threshold=0.5):
    model.eval()
    edge_label_index = data['patient', 'takes', 'medicine'].edge_label_index
    edge_label = data['patient', 'takes', 'medicine'].edge_label
    pred = model(data, edge_label_index).sigmoid()

    auc = roc_auc_score(edge_label.cpu(), pred.cpu())
    pred_binary = (pred > threshold).int().cpu().numpy()
    jaccard = jaccard_score(edge_label.cpu().numpy(), pred_binary)

    return auc, jaccard

# ---------------- Main ----------------
def main(args):
    # Load data
    data, mappings = load_and_process_data()

    # Split dataset
    transform = RandomLinkSplit(
        num_val=0.1, num_test=0.1,
        edge_types=('patient', 'takes', 'medicine'),
        rev_edge_types=('medicine', 'taken_by', 'patient'),
    )
    train_data, val_data, test_data = transform(data)

    # Model setup
    model = GNNModel(hidden_channels=64, out_channels=64, metadata=data.metadata(), data=data).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in range(1, args.epochs + 1):
        loss, bce, bpr = train(model, train_data, optimizer, loss_fn, alpha=0.5)
        val_auc, val_jaccard = evaluate(model, val_data)
        print(f"Epoch {epoch:02d} | Loss={loss:.4f} (BCE={bce:.4f}, BPR={bpr:.4f}) "
              f"| Val AUC={val_auc:.4f}, Val Jaccard={val_jaccard:.4f}")

    # Final Test
    test_auc, test_jaccard = evaluate(model, test_data)
    print(f"\nFinal Test AUC: {test_auc:.4f}")
    print(f"Final Test Jaccard: {test_jaccard:.4f}")

# ---------------- Entry Point ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    main(args)
