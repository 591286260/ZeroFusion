import os
import csv
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, TransformerConv

# ---------- Graph and Feature Preparation ----------

def build_graph_from_interactions(interaction_csv):
    df = pd.read_csv(interaction_csv, header=None, names=['circRNA', 'miRNA'])
    G = nx.Graph()
    for _, row in df.iterrows():
        circ, mir = row['circRNA'], row['miRNA']
        G.add_node(circ, type='circRNA')
        G.add_node(mir, type='miRNA')
        G.add_edge(circ, mir)
    return G

def load_node_features(feature_csv):
    df = pd.read_csv(feature_csv, header=None)
    df.columns = ['id'] + [f'feature_{i}' for i in range(1, df.shape[1])]
    return df.set_index('id').T.to_dict('list')

def convert_to_pyg_data(G, node_features):
    node_to_idx = {node: i for i, node in enumerate(G.nodes)}
    edge_index = torch.tensor(
        [[node_to_idx[u], node_to_idx[v]] for u, v in G.edges],
        dtype=torch.long
    ).t().contiguous()

    feature_dim = len(next(iter(node_features.values())))
    x = torch.zeros((len(G.nodes), feature_dim))
    for node, features in node_features.items():
        if node in node_to_idx:
            x[node_to_idx[node]] = torch.tensor(features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index), node_to_idx

# ---------- Graph Transformer Model ----------

class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transformer = TransformerConv(in_channels, 64, heads=1)
        self.gcn1 = GCNConv(64, 64)
        self.gcn2 = GCNConv(64, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.transformer(x, edge_index))
        x = torch.relu(self.gcn1(x, edge_index))
        x = self.gcn2(x, edge_index)
        return x

# ---------- Embedding Saving ----------

def save_node_embeddings(embeddings, index_to_node, output_path):
    df = pd.DataFrame(embeddings, index=index_to_node)
    df.to_csv(output_path, header=False)

# ---------- ID to Feature Conversion ----------

def read_feature_file(file_path):
    feature_dict = {}
    with open(file_path, 'r', encoding='gbk') as f:
        reader = csv.reader(f)
        for row in reader:
            feature_dict[row[0]] = row[1:]
    return feature_dict

def replace_ids_with_features(input_file, feature_dict, output_file):
    with open(input_file, 'r', encoding='gbk') as infile, \
         open(output_file, 'w', newline='', encoding='gbk') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for row in reader:
            id1, id2 = row
            feature1 = feature_dict.get(id1, [])
            feature2 = feature_dict.get(id2, [])
            writer.writerow(feature1 + feature2)

# ---------- Main Pipeline ----------

if __name__ == "__main__":
    # File paths
    interaction_path = '../dataset/CDM.csv'
    feature_path = 'Se_vector.csv'
    embedding_output_path = 'node_embeddings.csv'
    input_sample_path = '../../merged_dataset/positive_negative_samples.csv'
    sample_output_path = 'SampleFeature_best_params.csv'

    # Step 1: Build graph and prepare features
    graph = build_graph_from_interactions(interaction_path)
    features = load_node_features(feature_path)
    data, node_to_idx = convert_to_pyg_data(graph, features)

    # Step 2: Train graph model
    in_dim = data.x.shape[1]
    model = GraphTransformer(in_channels=in_dim, out_channels=in_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.006)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(390):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.x)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

    # Step 3: Save embeddings
    model.eval()
    embeddings = model(data).detach().numpy()
    index_to_node = list(graph.nodes)
    save_node_embeddings(embeddings, index_to_node, embedding_output_path)

    # Step 4: Replace sample IDs with trained features
    feature_dict = read_feature_file(embedding_output_path)
    replace_ids_with_features(input_sample_path, feature_dict, sample_output_path)
