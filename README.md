# Safe Medicine Recommendation via Heterogeneous GNN

This project is a Python implementation of the "Safe Medicine Recommendation" (SMR) framework using a modern Graph Neural Network (GNN) approach with PyTorch Geometric.

The model constructs a heterogeneous graph from multiple data sources to recommend medications for patients based on their diagnoses, while aiming to minimize adverse Drug-Drug Interactions (DDIs).

## 📚 Data Sources
- **MIMIC-III**: For patient-diagnosis and patient-prescription information.
- **ICD-9 Ontology**: For creating a disease hierarchy.
- **DrugBank**: For drug-drug interaction data.

## ⚙️ Setup

1.  **Clone the repository:**
    git clone <your-repo-url>
    cd safe-medicine-gnn

2.  **Set up the data:**
    - Download the MIMIC-III CSVs (`DIAGNOSES_ICD.csv`, `PRESCRIPTIONS.csv`) and place them inside the `data/MIMIC_III/` directory.
    - Download the DrugBank database (`drugbank_full_database.xml.zip`) and place it in the `data/` directory.

3.  **Install dependencies:**
    pip install -r requirements.txt


## ▶️ How to Run

    python src/train.py --epochs 50 --lr 0.001

