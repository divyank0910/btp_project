import pandas as pd
import numpy as np
import re
import xml.etree.ElementTree as ET
import zipfile
import torch
from torch_geometric.data import HeteroData
from collections import defaultdict
from difflib import SequenceMatcher
import pickle
import os

def normalize_drug_name(drug_name):
    name = str(drug_name).lower()
    name = re.sub(r'\(.*?\)|\d+(\.\d+)?\s?(mg|ml|mcg|units|%|l|g)\b', '', name)
    name = re.sub(r'\s+', ' ', name)
    suffixes = [' tablet', ' tablets', ' iv', ' injection', ' solution', ' capsule', 
                ' capsules', ' hcl', ' oral', ' er', ' sr', ' xl', ' cr', ' la']
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    name = re.sub(r'^(tab|cap|inj|sol)\s+', '', name)
    return name.strip()

def fuzzy_match_drug_names(drugbank_name, medicine_names, threshold=0.7):
    drugbank_name = drugbank_name.lower().strip()
    best_match = None
    best_score = 0
    for med_name in medicine_names:
        med_name_lower = med_name.lower().strip()
        if drugbank_name in med_name_lower or med_name_lower in drugbank_name:
            return med_name
        drugbank_words = set(drugbank_name.split())
        med_words = set(med_name_lower.split())
        if drugbank_words & med_words:
            word_overlap = len(drugbank_words & med_words) / len(drugbank_words | med_words)
            if word_overlap > 0.5:
                return med_name
        score = SequenceMatcher(None, drugbank_name, med_name_lower).ratio()
        if score > best_score and score > threshold:
            best_score = score
            best_match = med_name
    return best_match

def load_and_process_data():
    print("Starting Data Processing")
    print("Processing MIMIC-III diagnoses")
    try:
        diagnoses_df = pd.read_csv(r"C:\Users\user\Desktop\Python\SMR\data\MIMIC_III\DIAGNOSES_ICD.csv")
    except FileNotFoundError:
        print("data/MIMIC_III/DIAGNOSES_ICD.csv not found. Using dummy data.")
        diagnoses_df = pd.DataFrame({
            'subject_id': ['101', '101', '102', '103', '103', '104'],
            'icd9_code': ['4019', '25000', '4280', '41401', 'V4582', '4280']
        })
    diagnoses_df.dropna(subset=['subject_id', 'icd9_code'], inplace=True)
    diagnoses_df = diagnoses_df.astype({'subject_id': str, 'icd9_code': str})
    unique_patients = diagnoses_df['subject_id'].unique()
    patient_to_id = {p: i for i, p in enumerate(unique_patients)}
    unique_diseases = diagnoses_df['icd9_code'].unique()
    disease_to_id = {d: i for i, d in enumerate(unique_diseases)}
    id_to_disease = {i: d for d, i in disease_to_id.items()}
    patient_disease_edges_src = diagnoses_df['subject_id'].map(patient_to_id).values
    patient_disease_edges_dst = diagnoses_df['icd9_code'].map(disease_to_id).values
    patient_disease_edge_index = np.vstack([patient_disease_edges_src, patient_disease_edges_dst])
    print(f"Found {len(patient_to_id)} patients and {len(disease_to_id)} unique diseases.")
    print("Processing MIMIC-III prescriptions")
    try:
        prescriptions_df = pd.read_csv(r"C:\Users\user\Desktop\Python\SMR\data\MIMIC_III\PRESCRIPTIONS.csv", low_memory=False)
    except FileNotFoundError:
        print("data/MIMIC_III/PRESCRIPTIONS.csv not found. Using dummy data.")
        prescriptions_df = pd.DataFrame({
            'subject_id': ['101', '101', '102', '103', '104'],
            'drug': ['Aspirin 81 mg Tablet', 'Metformin 500mg', 'Furosemide 20mg IV', 'Warfarin', 'Aspirin']
        })
    prescriptions_df['subject_id'] = prescriptions_df['subject_id'].astype(str)
    prescriptions_df = prescriptions_df[prescriptions_df['subject_id'].isin(patient_to_id.keys())]
    prescriptions_df.dropna(subset=['subject_id', 'drug'], inplace=True)
    prescriptions_df['drug_NORMALIZED'] = prescriptions_df['drug'].apply(normalize_drug_name)
    unique_medicines = prescriptions_df['drug_NORMALIZED'].unique()
    medicine_to_id = {m: i for i, m in enumerate(unique_medicines)}
    id_to_medicine = {i: m for m, i in medicine_to_id.items()}
    prescriptions_df = prescriptions_df[prescriptions_df['drug_NORMALIZED'].isin(medicine_to_id.keys())]
    patient_medicine_edges_src = prescriptions_df['subject_id'].map(patient_to_id).values
    patient_medicine_edges_dst = prescriptions_df['drug_NORMALIZED'].map(medicine_to_id).values
    patient_medicine_edge_index = np.vstack([patient_medicine_edges_src, patient_medicine_edges_dst])
    print(f"Found {len(medicine_to_id)} unique normalized medicines.")
    print("Building disease hierarchy from ICD-9 using pyhealth")
    disease_hierarchy_edges = []
    try:
        from pyhealth.medcode import InnerMap
        print("Loading ICD9CM mappings...")
        icd9cm_map = InnerMap.load("ICD9CM")
        all_dataset_diseases = list(disease_to_id.keys())
        hierarchy_count = 0
        for disease_code_str in all_dataset_diseases:
            try:
                ancestors = icd9cm_map.get_ancestors(disease_code_str)
                if ancestors and len(ancestors) > 0:
                    for ancestor in reversed(ancestors):
                        if ancestor in disease_to_id and ancestor != disease_code_str:
                            child_id = disease_to_id[disease_code_str]
                            parent_id = disease_to_id[ancestor]
                            disease_hierarchy_edges.append([parent_id, child_id])
                            hierarchy_count += 1
                            break
            except Exception as e:
                continue
        print(f"Created {hierarchy_count} disease hierarchy edges using pyhealth.")
    except (ImportError, Exception) as e:
        print(f"Could not build disease hierarchy with pyhealth: {e}")
        print("Creating manual disease hierarchy based on ICD-9 structure...")
        category_groups = defaultdict(list)
        subcategory_groups = defaultdict(list)
        for disease_code in disease_to_id.keys():
            clean_code = disease_code.strip()
            if clean_code.startswith(('V', 'E')):
                continue
            if len(clean_code) >= 3:
                try:
                    if clean_code[:3].replace('.', '').isdigit():
                        category = clean_code[:3]
                        category_groups[category].append(clean_code)
                        if len(clean_code) >= 4:
                            subcategory = clean_code[:4]
                            subcategory_groups[subcategory].append(clean_code)
                except:
                    continue
        hierarchy_count = 0
        for category, codes in category_groups.items():
            if len(codes) > 1:
                codes_sorted = sorted(codes, key=len)
                parent_code = codes_sorted[0]
                parent_id = disease_to_id[parent_code]
                for child_code in codes_sorted[1:]:
                    if child_code != parent_code:
                        child_id = disease_to_id[child_code]
                        disease_hierarchy_edges.append([parent_id, child_id])
                        hierarchy_count += 1
        print(f"Created {hierarchy_count} disease hierarchy edges manually.")
    if disease_hierarchy_edges:
        disease_hierarchy_edge_index = np.array(disease_hierarchy_edges).T.astype(np.int64)
    else:
        disease_hierarchy_edge_index = np.empty((2, 0), dtype=np.int64)
    print(f"Final disease hierarchy edges: {disease_hierarchy_edge_index.shape[1]}")
    print("Extracting DDIs from DrugBank")
    ddi_edges = []
    drugbank_id_to_name = {}
    try:
        with zipfile.ZipFile(r"C:\Users\user\Desktop\Python\SMR\data\DrugBank\drugbank_full_database.zip", "r") as zf:
            with zf.open('drugbank_full_database.xml') as xml_file:
                print("Parsing DrugBank XML...")
                tree = ET.parse(xml_file)
                root = tree.getroot()
                drug_count = 0
                for drug_elem in root.findall(".//{http://www.drugbank.ca}drug"):
                    drugbank_id_elem = drug_elem.find('.//{http://www.drugbank.ca}drugbank-id[@primary="true"]')
                    name_elem = drug_elem.find('.//{http://www.drugbank.ca}name')
                    if drugbank_id_elem is None or name_elem is None:
                        continue
                    drugbank_id = drugbank_id_elem.text
                    name = normalize_drug_name(name_elem.text)
                    drugbank_id_to_name[drugbank_id] = name
                    drug_count += 1
                    interactions = drug_elem.find('.//{http://www.drugbank.ca}drug-interactions')
                    if interactions is not None:
                        for interaction in interactions.findall('.//{http://www.drugbank.ca}drug-interaction'):
                            interactant_id_elem = interaction.find('.//{http://www.drugbank.ca}drugbank-id')
                            if interactant_id_elem is not None:
                                ddi_edges.append((drugbank_id, interactant_id_elem.text))
                print(f"Loaded {drug_count} drugs from DrugBank with {len(ddi_edges)} interactions")
    except FileNotFoundError:
        print("DrugBank file not found. Using dummy DDI data.")
        drugbank_id_to_name = {
            'DB00945': 'aspirin', 
            'DB00682': 'warfarin',
            'DB00492': 'fosinopril',
            'DB00390': 'digoxin',
            'DB00641': 'simvastatin'
        }
        ddi_edges = [
            ('DB00945', 'DB00682'),
            ('DB00492', 'DB00390'),
            ('DB00682', 'DB00641'),
        ]
    print(f"Sample DrugBank drugs: {list(drugbank_id_to_name.values())[:10]}")
    print(f"Sample medicine names: {list(medicine_to_id.keys())[:10]}")
    medicine_name_to_internal_id = {}
    matches_found = 0
    print("Matching DrugBank drugs to MIMIC medicines...")
    for drugbank_name in drugbank_id_to_name.values():
        matched_name = fuzzy_match_drug_names(drugbank_name, list(medicine_to_id.keys()))
        if matched_name:
            medicine_name_to_internal_id[drugbank_name] = medicine_to_id[matched_name]
            matches_found += 1
            if matches_found <= 10:
                print(f"  Matched '{drugbank_name}' -> '{matched_name}'")
    print(f"Found {matches_found} drug name matches out of {len(drugbank_id_to_name)} DrugBank drugs")
    ddi_edge_index_list = []
    valid_interactions = 0
    for id1, id2 in ddi_edges:
        name1 = drugbank_id_to_name.get(id1)
        name2 = drugbank_id_to_name.get(id2)
        if name1 and name2:
            internal_id1 = medicine_name_to_internal_id.get(name1)
            internal_id2 = medicine_name_to_internal_id.get(name2)
            if internal_id1 is not None and internal_id2 is not None:
                ddi_edge_index_list.append([internal_id1, internal_id2])
                ddi_edge_index_list.append([internal_id2, internal_id1])
                valid_interactions += 1
    if ddi_edge_index_list:
        ddi_edge_index = np.array(ddi_edge_index_list).T.astype(np.int64)
    else:
        ddi_edge_index = np.empty((2, 0), dtype=np.int64)
    print(f"Created {valid_interactions} unique DDI relationships ({ddi_edge_index.shape[1]} total edges)")
    if ddi_edge_index.shape[1] == 0 and len(medicine_to_id) > 1:
        print("No DDI edges found. Adding dummy edges for testing...")
        med_ids = list(range(min(4, len(medicine_to_id))))
        dummy_edges = []
        for i in range(len(med_ids)):
            for j in range(i+1, len(med_ids)):
                dummy_edges.extend([[med_ids[i], med_ids[j]], [med_ids[j], med_ids[i]]])
        ddi_edge_index = np.array(dummy_edges).T.astype(np.int64)
        print(f"Added {len(dummy_edges)} dummy DDI edges")
    print("Assembling HeteroData object")
    data = HeteroData()
    data['patient'].num_nodes = len(patient_to_id)
    data['disease'].num_nodes = len(disease_to_id)
    data['medicine'].num_nodes = len(medicine_to_id)
    data['patient', 'diagnosed_with', 'disease'].edge_index = torch.from_numpy(patient_disease_edge_index).to(torch.long)
    data['patient', 'takes', 'medicine'].edge_index = torch.from_numpy(patient_medicine_edge_index).to(torch.long)
    data['medicine', 'interacts_with', 'medicine'].edge_index = torch.from_numpy(ddi_edge_index).to(torch.long)
    data['disease', 'parent_of', 'disease'].edge_index = torch.from_numpy(disease_hierarchy_edge_index).to(torch.long)
    data['disease', 'affects', 'patient'].edge_index = data['patient', 'diagnosed_with', 'disease'].edge_index.flip([0])
    data['medicine', 'taken_by', 'patient'].edge_index = data['patient', 'takes', 'medicine'].edge_index.flip([0])
    data['disease', 'child_of', 'disease'].edge_index = data['disease', 'parent_of', 'disease'].edge_index.flip([0])
    print("Data Processing Finished")
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

if __name__ == "__main__":
    data, mappings = load_and_process_data()
    os.makedirs(r"C:\Users\user\Desktop\Python\SMR\data\Processed", exist_ok=True)
    torch.save(data, r"C:\Users\user\Desktop\Python\SMR\data\Processed\hetero_data.pt")
    print("HeteroData object saved to data/Processed/hetero_data.pt")
    with open(r"C:\Users\user\Desktop\Python\SMR\data\Processed\mappings.pkl", "wb") as f:
        pickle.dump(mappings, f)
    print("Mappings saved to data/Processed/mappings.pkl")
