import argparse
import torch
import numpy as np
from collections import defaultdict

from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score

from data_processing import load_and_process_data
from model import SMR_Model

# --- Training & Evaluation Functions ---

def train(model, train_data, optimizer, loss_fn):
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
    model.eval()
    pred = model(data_split, data_split['patient', 'takes', 'medicine'].edge_label_index)
    target = data_split['patient', 'takes', 'medicine'].edge_label
    pred_probs = pred.sigmoid()
    return roc_auc_score(target.cpu().numpy(), pred_probs.cpu().numpy())

# --- Metrics Calculation ---

@torch.no_grad()
def get_top_k_recs(model, test_data, z_dict, patient_internal_id, k=10):
    model.eval()
    patient_emb = z_dict['patient'][patient_internal_id]
    medicine_embs = z_dict['medicine']
    patient_emb_replicated = patient_emb.repeat(medicine_embs.size(0), 1)
    scores = model.predictor(patient_emb_replicated, medicine_embs)
    return torch.topk(scores, k).indices.cpu().numpy()

def calculate_jaccard_similarity(model, test_data, true_recs_by_patient):
    z_dict = model.get_all_embeddings(test_data)
    jaccard_scores =
    for patient_id, true_medicines in true_recs_by_patient.items():
        k = len(true_medicines)
        if k == 0: continue
        
        recommended_medicines = set(get_top_k_recs(model, test_data, z_dict, patient_id, k=k))
        intersection = len(recommended_medicines.intersection(true_medicines))
        union = len(recommended_medicines.union(true_medicines))
        
        if union > 0:
            jaccard_scores.append(intersection / union)
            
    return np.mean(jaccard_scores) if jaccard_scores else 0.0

def calculate_ddi_rate(model, test_data, true_recs_by_patient, ddi_edge_index):
    z_dict = model.get_all_embeddings(test_data)
    ddi_set = set()
    for i in range(ddi_edge_index.shape[1]):
        u, v = ddi_edge_index[:, i]
        ddi_set.add(tuple(sorted((u, v))))

    total_patients, patients_with_ddi = 0, 0
    for patient_id, true_medicines in true_recs_by_patient.items():
        k = len(true_medicines)
        if k < 2: continue
        
        total_patients += 1
        recs = get_top_k_recs(model, test_data, z_dict, patient_id, k=k)
        
        has_ddi = False
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                med1, med2 = recs[i], recs[j]
                if tuple(sorted((med1, med2))) in ddi_set:
                    has_ddi = True
                    break
            if has_ddi: break
        
        if has_ddi:
            patients_with_ddi += 1
            
    return (patients_with_ddi / total_patients) if total_patients > 0 else 0.0

# --- Inference Function ---

def recommend_for_new_patient(model, test_data, mappings, diagnosis_codes, k=5):
    model.eval()
    z_dict = model.get_all_embeddings(test_data)
    disease_embs = z_dict['disease']
    medicine_embs = z_dict['medicine']
    
    disease_ids = [mappings['disease_to_id'].get(code) for code in diagnosis_codes if code in mappings['disease_to_id']]
    if not disease_ids:
        print("Warning: None of the provided diagnosis codes were found in the dataset.")
        return
        
    device = next(model.parameters()).device
    patient_disease_embs = disease_embs[torch.tensor(disease_ids, device=device)]
    patient_emb = patient_disease_embs.mean(dim=0)
    
    patient_emb_replicated = patient_emb.repeat(medicine_embs.size(0), 1)
    scores = model.predictor(patient_emb_replicated, medicine_embs)
    top_k_indices = torch.topk(scores, k).indices.cpu().numpy()
    
    return [mappings['id_to_medicine'].get(idx) for idx in top_k_indices]

# --- Main Execution Block ---

def main():
    parser = argparse.ArgumentParser(description='SMR GNN Implementation')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Number of hidden channels in the GNN')
    parser.add_argument('--out_channels', type=int, default=64, help='Number of output channels from the GNN encoder')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    args = parser.parse_args()

    # Load and process data
    data, mappings = load_and_process_data()

    # Split data for link prediction
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=False,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
        edge_types=[('patient', 'takes', 'medicine')],
        rev_edge_types=[('medicine', 'taken_by', 'patient')],
    )
    train_data, val_data, test_data = transform(data)

    # Setup model and training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SMR_Model(data, args.hidden_channels, args.out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_data, val_data, test_data = train_data.to(device), val_data.to(device), test_data.to(device)

    # Training loop
    print("\n--- Starting Model Training ---")
    for epoch in range(1, args.epochs + 1):
        loss = train(model, train_data, optimizer, loss_fn)
        val_auc = test(model, val_data)
        print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}")

    print("\n--- Training Finished ---")

    # Final evaluation
    print("\n--- Final Model Evaluation ---")
    test_auc = test(model, test_data)
    
    test_pos_edges = test_data['patient', 'takes', 'medicine'].edge_label_index[:, test_data['patient', 'takes', 'medicine'].edge_label == 1]
    true_recs_by_patient = defaultdict(set)
    for i in range(test_pos_edges.shape[1]):
        true_recs_by_patient[test_pos_edges[0, i].item()].add(test_pos_edges[1, i].item())

    jaccard_score = calculate_jaccard_similarity(model, test_data, true_recs_by_patient)
    ddi_rate = calculate_ddi_rate(model, test_data, true_recs_by_patient, mappings['ddi_edge_index'])
    
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Jaccard Score: {jaccard_score:.4f}")
    print(f"DDI Rate: {ddi_rate * 100:.2f}%")

    # Inference example
    print("\n--- Inference Example ---")
    new_patient_diagnoses = ['4280', '4019'] # Congestive heart failure, Hypertension
    recommendations = recommend_for_new_patient(model, test_data, mappings, new_patient_diagnoses, k=5)
    print(f"Recommendations for patient with diagnoses {new_patient_diagnoses}:")
    if recommendations:
        for med in recommendations:
            print(f"- {med}")
    else:
        print("Could not generate recommendations.")

if __name__ == '__main__':
    main()
