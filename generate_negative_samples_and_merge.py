import os
import random
import csv
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

def read_fasta(fasta_path):
    circ_data = {}
    with open(fasta_path, 'r') as file:
        current_id = None
        for line in file:
            if line.startswith('>'):
                current_id = line.strip()[1:]
            else:
                circ_data[current_id] = line.strip()
    return circ_data

def read_xlsx(xlsx_path):
    mirna_data = {}
    df = pd.read_excel(xlsx_path, header=None)
    for _, row in df.iterrows():
        mirna_id = row[0]
        location = row[1]
        mirna_data.setdefault(mirna_id, []).append(location)
    return mirna_data

def read_positive_csv(csv_path):
    df = pd.read_csv(csv_path, header=None)
    return list(zip(df[0], df[1]))

def generate_negative_samples(circ_data, mirna_data, positive_samples):
    all_pairs = [(c, m) for c in circ_data for m in mirna_data]
    positive_set = set(positive_samples)
    candidate_set = set(all_pairs) - positive_set

    filtered_negatives = []
    for circ_id, mirna_id in tqdm(candidate_set, desc="Filtering Negatives"):
        circ_loc = circ_data[circ_id]
        mirna_locs = mirna_data[mirna_id]
        if all(loc not in circ_loc for loc in mirna_locs):
            filtered_negatives.append((circ_id, mirna_id))

    if len(filtered_negatives) < len(positive_samples):
        raise ValueError("Not enough negative samples after filtering.")

    return random.sample(filtered_negatives, len(positive_samples))

def save_csv(data, output_path, headers=None):
    df = pd.DataFrame(data, columns=headers if headers else None)
    df.to_csv(output_path, index=False, header=bool(headers))

def read_csv_to_list(path):
    with open(path, 'r') as f:
        return list(csv.reader(f))

def merge_and_save(pos_file, neg_file, output_file):
    pos_data = read_csv_to_list(pos_file)
    neg_data = read_csv_to_list(neg_file)

    print(f"Positive samples: {len(pos_data)}")
    print(f"Negative samples: {len(neg_data)}")

    merged = np.vstack((pos_data, neg_data))
    print(f"Merged shape: {merged.shape}")

    write_csv(merged.tolist(), output_file)

def write_csv(data, output_file):
    with open(output_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

if __name__ == "__main__":
    circ_fasta_path = "../../dataset/circRNA_subcellular.fasta"
    mirna_xlsx_path = "../../dataset/miRNA_subcellular.xlsx"
    pos_sample_path = "../../dataset/valid_CMI.csv"
    neg_output_path = "../../NegativeSample/NegativeSample.csv"
    final_output_path = "CMI_Positive_Negative_Combined.csv"

    circ_data = read_fasta(circ_fasta_path)
    mirna_data = read_xlsx(mirna_xlsx_path)
    pos_samples = read_positive_csv(pos_sample_path)

    neg_samples = generate_negative_samples(circ_data, mirna_data, pos_samples)
    save_csv(neg_samples, neg_output_path)

    merge_and_save(pos_sample_path, neg_output_path, final_output_path)
