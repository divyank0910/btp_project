import pandas as pd
import numpy as np
import re
import xml.etree.ElementTree as ET
import zipfile
import torch
from torch_geometric.data import HeteroData

def load_and_process_data():
    """
    Loads and processes MIMIC-III, ICD-9, and DrugBank data to create a
    heterogeneous graph.
    """
    print("--- Starting Data Processing ---")

    # --- 1.1. Process MIMIC-III Diagnoses ---
    print("Processing MIMIC-III diagnoses...")
    try:
        diagnoses_df = pd.read_csv('data/MIMIC_III/DIAGNOSES_ICD.csv')
    except FileNotFoundError:
        print("data/MIMIC_III/DIAGNOSES_ICD.csv not found. Please ensure you have access and have placed it in the correct directory.")
        # Create dummy dataframe for demonstration if file not found
        diagnoses_df = pd.DataFrame({
            'SUBJECT_ID': ,
            'ICD9_CODE': ['4019', '25000', '4280', '41401', 'V4582', '4280']
        })
    diagnoses_df.dropna(subset=, inplace=True)
    diagnoses_df = diagnoses_df.astype({'SUBJECT_ID': str, 'ICD9_CODE': str})

    unique_patients = diagnoses_df.unique()
    patient_to_id = {p: i for i, p in enumerate(unique_patients)}

    unique_diseases = diagnoses_df.unique()
    disease_to_id = {d: i for i, d in enumerate(unique_diseases)}
    id_to_disease = {i: d for d, i in disease_to_id.items()}

    patient_disease_edges_src = diagnoses_df.map(patient_to_id).values
    patient_disease_edges_dst = diagnoses_df.map(disease_to_id).values
    patient_disease_edge_index = np.vstack([patient_disease_edges_src, patient_disease_edges_dst])

    print(f"Found {len(patient_to_id)} patients and {len(disease_to_id)} unique diseases.")

    # --- 1.2. Process MIMIC-III Prescriptions ---
    print("Processing MIMIC-III prescriptions...")
    try:
        prescriptions_df = pd.read_csv('data/MIMIC_III/PRESCRIPTIONS.csv', low_memory=False)
    except FileNotFoundError:
        print("data/MIMIC_III/PRESCRIPTIONS.csv not found. Using dummy data.")
        prescriptions_df = pd.DataFrame({
            'SUBJECT_ID': ,
            'DRUG':
        })

    prescriptions_df = prescriptions_df.isin(patient_to_id.keys())]
    prescriptions_df.dropna(subset=, inplace=True)

    def normalize_drug_name(drug_name):
        name = str(drug_name).lower()
        name = re.sub(r'\(.*?\)|\d+(\.\d+)?\s?(mg|ml|mcg|units|%|l)\b', '', name)
        suffixes = [' tablet', ' iv', ' injection', ' solution', ' capsule', ' hcl', ' oral']
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        return name.strip()

    prescriptions_df = prescriptions_df.apply(normalize_drug_name)
    
    unique_medicines = prescriptions_df.unique()
    medicine_to_id = {m: i for i, m in enumerate(unique_medicines)}
    id_to_medicine = {i: m for m, i in medicine_to_id.items()}

    prescriptions_df = prescriptions_df.isin(medicine_to_id.keys())]
    patient_medicine_edges_src = prescriptions_df.map(patient_to_id).values
    patient_medicine_edges_dst = prescriptions_df.map(medicine_to_id).values
    patient_medicine_edge_index = np.vstack([patient_medicine_edges_src, patient_medicine_edges_dst])

    print(f"Found {len(medicine_to_id)} unique normalized medicines.")

    # --- 1.3. Build Disease Hierarchy from ICD-9 ---
    print("Building disease hierarchy from ICD-9 using pyhealth...")
    disease_hierarchy_edges =
    try:
        from pyhealth.medcode import InnerMap
        icd9cm_map = InnerMap.load("ICD9CM")
        all_dataset_diseases = list(disease_to_id.keys())
        for disease_code_str in all_dataset_diseases:
            try:
                ancestors = icd9cm_map.get_ancestors(disease_code_str)
                if ancestors:
                    parent_code = ancestors
                    if parent_code in disease_to_id:
                        child_id = disease_to_id[disease_code_str]
                        parent_id = disease_to_id[parent_code]
                        disease_hierarchy_edges.append([parent_id, child_id])
            except Exception:
                continue
    except (ImportError, Exception) as e:
        print(f"Could not build disease hierarchy via pyhealth: {e}. Using dummy data.")
        # Create dummy hierarchy if pyhealth fails or codes don't map
        if '4019' in disease_to_id and '390-459.99' in disease_to_id:
             disease_hierarchy_edges.append([disease_to_id['390-459.99'], disease_to_id['4019']])
    
    disease_hierarchy_edge_index = np.array(disease_hierarchy_edges).T if disease_hierarchy_edges else np.empty((2, 0), dtype=np.int64)
    print(f"Created {disease_hierarchy_edge_index.shape[1]} disease hierarchy edges.")

    # --- 1.4. Extract DDIs from DrugBank ---
    print("Extracting DDIs from DrugBank...")
    ddi_edges =
    drugbank_id_to_name = {}
    ns = '{http://www.drugbank.ca}'
    try:
        with zipfile.ZipFile('data/drugbank_full_database.xml.zip', 'r') as zf:
            with zf.open('full database.xml') as xml_file:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                for drug_elem in root.findall(f'{ns}drug'):
                    drugbank_id = drug_elem.find(f'{ns}drugbank-id[@primary="true"]').text
                    name = drug_elem.find(f'{ns}name').text.lower()
                    drugbank_id_to_name[drugbank_id] = name
                    interactions = drug_elem.find(f'{ns}drug-interactions')
                    if interactions:
                        for interaction in interactions.findall(f'{ns}drug-interaction'):
                            interactant_id = interaction.find(f'{ns}drugbank-id').text
                            description = interaction.find(f'{ns}description').text.lower()
                            if 'risk' in description or 'adverse' in description:
                                ddi_edges.append((drugbank_id, interactant_id))
    except FileNotFoundError:
        print("data/drugbank_full_database.xml.zip not found. Using dummy data.")
        drugbank_id_to_name = {'DB00945': 'aspirin', 'DB00047': 'warfarin'}
        ddi_edges =

    name_to_drugbank_id = {v: k for k, v in drugbank_id_to_name.items()}
    medicine_name_to_internal_id = {name: medicine_to_id.get(name) for name in name_to_drugbank_id.values()}
    
    ddi_edge_index_list =
    for id1, id2 in ddi_edges:
        name1, name2 = drugbank_id_to_name.get(id1), drugbank_id_to_name.get(id2)
        internal_id1, internal_id2 = medicine_name_to_internal_id.get(name1), medicine_name_to_internal_id.get(name2)
        if internal_id1 is not None and internal_id2 is not None:
            ddi_edge_index_list.append([internal_id1, internal_id2])
            ddi_edge_index_list.append([internal_id2, internal_id1])
    
    ddi_edge_index = np.array(ddi_edge_index_list).T if ddi_edge_index_list else np.empty((2, 0), dtype=np.int64)
    print(f"Created {ddi_edge_index.shape[1]} DDI edges.")

    # --- 1.5. Assemble HeteroData Object ---
    print("Assembling HeteroData object...")
    data = HeteroData()
    data['patient'].num_nodes = len(patient_to_id)
    data['disease'].num_nodes = len(disease_to_id)
    data['medicine'].num_nodes = len(medicine_to_id)

    data['patient', 'diagnosed_with', 'disease'].edge_index = torch.from_numpy(patient_disease_edge_index).to(torch.long)
    data['patient', 'takes', 'medicine'].edge_index = torch.from_numpy(patient_medicine_edge_index).to(torch.long)
    data['medicine', 'interacts_with', 'medicine'].edge_index = torch.from_numpy(ddi_edge_index).to(torch.long)
    data['disease', 'parent_of', 'disease'].edge_index = torch.from_numpy(disease_hierarchy_edge_index).to(torch.long)

    # Add reverse edges
    data['disease', 'affects', 'patient'].edge_index = data['patient', 'diagnosed_with', 'disease'].edge_index[]
    data['medicine', 'taken_by', 'patient'].edge_index = data['patient', 'takes', 'medicine'].edge_index[]
    data['disease', 'child_of', 'disease'].edge_index = data['disease', 'parent_of', 'disease'].edge_index[]

    print("--- Data Processing Finished ---")
    print(data)
    
    mappings = {
        'patient_to_id': patient_to_id, 
        'disease_to_id': disease_to_id, 
        'id_to_disease': id_to_disease,
        'medicine_to_id': medicine_to_id, 
        'id_to_medicine': id_to_medicine,
        'ddi_edge_index': ddi_edge_index
    }
    return data, mappings
