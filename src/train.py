import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GATConv, to_hetero
from sklearn.metrics import roc_auc_score

# Import the data loading function from your data processing script
from src.data_processing import load_and_process_data

# --- 2. Model Definition ---

class GNNEncoder(nn.Module):
    """Graph Neural Network Encoder using GATConv layers."""
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class LinkPredictor(nn.Module):
    """Predicts the existence of a link based on node embeddings."""
    def __init__(self, in_channels):
        super().__init__()
        self.lin1 = nn.Linear(2 * in_channels, in_channels)
        self.lin2 = nn.Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        # Concatenate source and destination node embeddings
        x = torch.cat([z_src, z_dst], dim=-1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return x.squeeze(-1)

class SMR_Model(nn.Module):
    """The main model for Safe Medicine Recommendation."""
    def __init__(self, data, hidden_channels, out_channels):
        super().__init__()
        # Learnable embeddings for each node type
        self.patient_emb = nn.Embedding(data['patient'].num_nodes, hidden_channels)
        self.disease_emb = nn.Embedding(data['disease'].num_nodes, hidden_channels)
        self.medicine_emb = nn.Embedding(data['medicine'].num_nodes, hidden_channels)
        
        # Instantiate the GNN encoder and convert it to a heterogeneous model
        encoder = GNNEncoder(hidden_channels, out_channels)
        self.encoder = to_hetero(encoder, data.metadata(), aggr='sum')
        
        # Instantiate the link predictor
        self.predictor = LinkPredictor(out_channels)

    def forward(self, data, edge_label_index):
        # Get initial node features from embedding layers
        x_dict = {
            "patient": self.patient_emb.weight,
            "disease": self.disease_emb.weight,
            "medicine": self.medicine_emb.weight,
        }
        
        # Encode all nodes using the GNN
        z_dict = self.encoder(x_dict, data.edge_index_dict)
        
        # Get source (patient) and destination (medicine) embeddings for prediction
        src = z_dict['patient'][edge_label_index[0]]
        dst = z_dict['medicine'][edge_label_index[1]]
        
        return self.predictor(src, dst)

# --- 3. Training and Evaluation Functions ---

def train(model, train_data, optimizer, loss_fn):
    """Performs a single training step."""
    model.train()
    optimizer.zero_grad()
    pred = model(train_data, train_data['patient', 'takes', 'medicine'].edge_label_index)
    target = train_data['patient', 'takes', 'medicine'].edge_label
    loss = loss_fn(pred, target.float())
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(model, data_split):
    """Evaluates the model on a given data split."""
    model.eval()
    pred = model(data_split, data_split['patient', 'takes', 'medicine'].edge_label_index)
    target = data_split['patient', 'takes', 'medicine'].edge_label
    pred_probs = pred.sigmoid()
    return roc_auc_score(target.cpu().numpy(), pred_probs.cpu().numpy())

# --- 4. Advanced Evaluation Metrics ---

@torch.no_grad()
def get_top_k_recs(model, test_data, patient_internal_id, k=10):
    """Generates top-k medicine recommendations for a single patient."""
    model.eval()
    x_dict = {
        "patient": model.patient_emb.weight,
        "disease": model.disease_emb.weight,
        "medicine": model.medicine_emb.weight,
    }
    z_dict = model.encoder(x_dict, test_data.edge_index_dict)
    patient_emb = z_dict['patient'][patient_internal_id]
    medicine_embs = z_dict['medicine']
    patient_emb_replicated = patient_emb.repeat(medicine_embs.size(0), 1)
    scores = model.predictor(patient_emb_replicated, medicine_embs)
    return torch.topk(scores, k).indices.cpu().numpy()

def calculate_jaccard_similarity(model, test_data, true_recs_by_patient):
    """Calculates the average Jaccard similarity across all test patients."""
    jaccard_scores = []
    for patient_id, true_medicines in true_recs_by_patient.items():
        k = len(true_medicines)
        if k == 0: continue
        recommended_medicines = set(get_top_k_recs(model, test_data, patient_id, k=k))
        intersection = len(recommended_medicines.intersection(true_medicines))
        union = len(recommended_medicines.union(true_medicines))
        if union > 0:
            jaccard_scores.append(intersection / union)
    return np.mean(jaccard_scores) if jaccard_scores else 0.0

def calculate_ddi_rate(model, test_data, true_recs_by_patient, ddi_edge_index):
    """Calculates the rate of recommendations containing adverse DDIs."""
    ddi_set = set()
    for i in range(ddi_edge_index.shape[1]):
        u, v = ddi_edge_index[:, i]
        ddi_set.add(tuple(sorted((u, v))))
    
    total_patients, patients_with_ddi = 0, 0
    for patient_id, true_medicines in true_recs_by_patient.items():
        k = len(true_medicines)
        if k < 2: continue
        total_patients += 1
        recs = get_top_k_recs(model, test_data, patient_id, k=k)
        has_ddi = False
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                if tuple(sorted((recs[i], recs[j]))) in ddi_set:
                    has_ddi = True
                    break
            if has_ddi: break
        if has_ddi:
            patients_with_ddi += 1
    return (patients_with_ddi / total_patients) if total_patients > 0 else 0.0

# --- 5. Inference for New Patients ---
@torch.no_grad()
def recommend_for_new_patient(model, test_data, diagnosis_codes, mappings, k=5):
    """Generates recommendations for a new patient based on their diagnoses."""
    model.eval()
    
    # Get learned embeddings from the trained model
    x_dict = {
        "patient": model.patient_emb.weight,
        "disease": model.disease_emb.weight,
        "medicine": model.medicine_emb.weight,
    }
    z_dict = model.encoder(x_dict, test_data.edge_index_dict)
    disease_embs = z_dict['disease']
    medicine_embs = z_dict['medicine']
    
    # Map diagnosis codes to internal IDs
    disease_ids = [mappings['disease_to_id'].get(code) for code in diagnosis_codes if code in mappings['disease_to_id']]
    if not disease_ids:
        print("Warning: None of the provided diagnosis codes were found in the dataset.")
        return []
    
    # Create patient embedding by averaging their disease embeddings
    patient_disease_embs = disease_embs[torch.tensor(disease_ids, device=next(model.parameters()).device)]
    patient_emb = patient_disease_embs.mean(dim=0)
    
    patient_emb_replicated = patient_emb.repeat(medicine_embs.size(0), 1)
    
    scores = model.predictor(patient_emb_replicated, medicine_embs)
    top_k_indices = torch.topk(scores, k).indices.cpu().numpy()
    
    return [mappings['id_to_medicine'].get(idx) for idx in top_k_indices]

# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMR GNN Training and Evaluation')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Number of hidden channels')
    parser.add_argument('--out_channels', type=int, default=64, help='Number of output channels')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    args = parser.parse_args()

    # 1. Load and process data using the external script
    data, mappings = load_and_process_data()

    # 2. Split data for link prediction
    transform = RandomLinkSplit(
        num_val=0.1, num_test=0.1, is_undirected=False,
        add_negative_train_samples=True, neg_sampling_ratio=1.0,
        edge_types=[('patient', 'takes', 'medicine')],
        rev_edge_types=[('medicine', 'taken_by', 'patient')],
    )
    train_data, val_data, test_data = transform(data)

    # 3. Setup model, optimizer, and loss function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SMR_Model(data, args.hidden_channels, args.out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    # 4. Training loop
    print("\n--- Starting Model Training ---")
    for epoch in range(1, args.epochs + 1):
        loss = train(model, train_data, optimizer, loss_fn)
        val_auc = test(model, val_data)
        print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}")

    print("\n--- Training Finished ---")

    # 5. Final evaluation on the test set
    print("\n--- Final Model Evaluation ---")
    test_auc = test(model, test_data)
    
    # Prepare ground truth for Jaccard and DDI calculations
    test_pos_edges = test_data['patient', 'takes', 'medicine'].edge_label_index[:, test_data['patient', 'takes', 'medicine'].edge_label == 1]
    true_recs_by_patient = defaultdict(set)
    for i in range(test_pos_edges.shape[1]):
        patient_id = test_pos_edges[0, i].item()
        medicine_id = test_pos_edges[1, i].item()
        true_recs_by_patient[patient_id].add(medicine_id)

    jaccard_score = calculate_jaccard_similarity(model, test_data, true_recs_by_patient)
    ddi_rate = calculate_ddi_rate(model, test_data, true_recs_by_patient, mappings['ddi_edge_index'])
    
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Jaccard Score: {jaccard_score:.4f}")
    print(f"DDI Rate: {ddi_rate * 100:.4f}%")

    # 6. Inference example for a new patient
    print("\n--- Inference Example ---")
    new_patient_diagnoses = ['4280', '4019'] # Congestive heart failure, Hypertension
    recommendations = recommend_for_new_patient(model, test_data, new_patient_diagnoses, mappings, k=5)
    print(f"Recommendations for patient with diagnoses {new_patient_diagnoses}:")
    if recommendations:
        for med in recommendations:
            print(f"- {med}")

