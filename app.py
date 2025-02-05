import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np

###########################################
# 1. LOAD CSV AND PREPARE THE DATA
###########################################

# Change this path to your actual code error dataset.
csv_path = "code_errors.csv"
df = pd.read_csv(csv_path)

# --- Build Unique Node Sets ---
# Use code snippets as one node set and error types as the other.
code_snippet_ids = df["code_snippet_id"].unique()
error_type_ids = df["error_type_id"].unique()

num_code_snippets = len(code_snippet_ids)
num_error_types = len(error_type_ids)
total_nodes = num_code_snippets + num_error_types

# Create mapping dictionaries:
# For code snippets: assign indices 0 to num_code_snippets-1.
code_snippet_id_to_index = {cid: i for i, cid in enumerate(code_snippet_ids)}
# For error types: assign indices num_code_snippets to total_nodes-1.
error_type_id_to_index = {eid: i + num_code_snippets for i, eid in enumerate(error_type_ids)}

# --- Build Graph Edges ---
# Each row in the CSV corresponds to an occurrence of an error in a code snippet.
source_nodes = []  # code snippet nodes (indices)
target_nodes = []  # error type nodes (indices)
for _, row in df.iterrows():
    cid = row["code_snippet_id"]
    eid = row["error_type_id"]
    if cid in code_snippet_id_to_index and eid in error_type_id_to_index:
        source_nodes.append(code_snippet_id_to_index[cid])
        target_nodes.append(error_type_id_to_index[eid])

# For an undirected graph, add both directions.
edge_index = torch.tensor(
    [source_nodes + target_nodes, target_nodes + source_nodes], dtype=torch.long
)

# --- Create Node Features ---
# We choose a set of features that describe code snippets (e.g., length, complexity, etc.).
code_feature_cols = [
    "code_length",
    "num_loops",
    "num_functions",
    "num_variables",
    "num_imports",
]
num_features = len(code_feature_cols)

# Instead of initializing error types with zeros, we initialize them with random vectors.
code_features = torch.randn((num_code_snippets, num_features), dtype=torch.float)

# Build a DataFrame of unique error types (first occurrence per error type).
error_type_df = df.drop_duplicates("error_type_id").set_index("error_type_id")

# Initialize error type feature matrix.
error_type_features = torch.zeros((num_error_types, num_features), dtype=torch.float)
for eid in error_type_ids:
    if eid in error_type_df.index:
        try:
            feats = [float(error_type_df.loc[eid][col]) for col in code_feature_cols]
        except Exception as e:
            feats = [0.0] * num_features
        error_type_features[error_type_ids.tolist().index(eid)] = torch.tensor(
            feats, dtype=torch.float
        )
    else:
        error_type_features[error_type_ids.tolist().index(eid)] = torch.zeros(num_features)

# OPTIONAL: Normalize features (min-max scaling)
min_vals = error_type_features.min(dim=0)[0]
max_vals = error_type_features.max(dim=0)[0]
range_vals = max_vals - min_vals
range_vals[range_vals == 0] = 1.0
error_type_features = (error_type_features - min_vals) / range_vals

# Concatenate code snippet and error type features to create node features.
x = torch.cat([code_features, error_type_features], dim=0)

# Create the PyG Data object.
data = Data(x=x, edge_index=edge_index)
print(
    f"Graph built: {total_nodes} nodes ({num_code_snippets} code snippets, {num_error_types} error types), "
    f"{edge_index.shape[1]//2} undirected edges."
)

###########################################
# 2. DEFINE THE GNN MODEL
###########################################

class GNNDebugger(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNDebugger, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.5)
        x = self.conv2(x, edge_index)
        return x

in_channels = num_features
hidden_channels = 128
out_channels = 64

model = GNNDebugger(in_channels, hidden_channels, out_channels)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

###########################################
# 3. TRAIN THE GNN
###########################################

# Prepare positive edges (only one direction is used for training)
pos_code_snippet_nodes = torch.tensor(source_nodes, dtype=torch.long)
pos_error_type_nodes = torch.tensor(target_nodes, dtype=torch.long)
num_pos_edges = pos_code_snippet_nodes.shape[0]

num_epochs = 100
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    embeddings = model(data.x, data.edge_index)

    # Compute positive scores (dot product between code snippet and error type embeddings)
    pos_code_snippet_emb = embeddings[pos_code_snippet_nodes]
    pos_error_type_emb = embeddings[pos_error_type_nodes]
    pos_scores = (pos_code_snippet_emb * pos_error_type_emb).sum(dim=1)
    pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-15).mean()

    # Negative sampling: sample random code snippet-error type pairs.
    neg_code_snippet_nodes = torch.randint(0, num_code_snippets, (num_pos_edges,))
    neg_error_type_nodes = torch.randint(num_code_snippets, total_nodes, (num_pos_edges,))
    neg_code_snippet_emb = embeddings[neg_code_snippet_nodes]
    neg_error_type_emb = embeddings[neg_error_type_nodes]
    neg_scores = (neg_code_snippet_emb * neg_error_type_emb).sum(dim=1)
    neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-15).mean()

    loss = pos_loss + neg_loss
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}")

###########################################
# 4. INTERACTIVE ERROR DETECTION
###########################################

# Switch to evaluation mode and compute final embeddings.
model.eval()
with torch.no_grad():
    final_embeddings = model(data.x, data.edge_index)

# Separate code snippet and error type embeddings.
code_snippet_embeddings = final_embeddings[:num_code_snippets]  # indices 0 .. num_code_snippets-1
error_type_embeddings = final_embeddings[
    num_code_snippets:
]  # indices num_code_snippets .. total_nodes-1

# Build an inverse mapping from error type node index (global index) to error_type_id.
inv_error_type_map = {v: k for k, v in error_type_id_to_index.items()}

def get_code_snippet_node_index_by_name(query_name):
    """
    Given a code snippet ID, return (code_snippet_id, global_node_index) if found.
    The search is case-insensitive and expects an exact match.
    """
    matches = df[df["code_snippet_id"].str.lower() == query_name.lower()]
    if matches.empty:
        return None, None
    else:
        code_snippet_id = matches["code_snippet_id"].iloc[0]
        node_index = code_snippet_id_to_index.get(code_snippet_id)
        return code_snippet_id, node_index


print("\n--- Interactive Error Detection ---")
print(
    "Type a code snippet ID (exactly as in the dataset) to get 10 similar error type recommendations."
)
print("Type 'quit' to exit.\n")

while True:
    query_name = input("Enter a code snippet ID: ").strip()
    if query_name.lower() == "quit":
        print("Exiting error detection system.")
        break

    code_snippet_id, node_index = get_code_snippet_node_index_by_name(query_name)
    if node_index is None:
        print("Code snippet not found. Please try again.\n")
        continue

    # Retrieve the embedding of the query code snippet.
    code_embedding = final_embeddings[node_index]

    # Compute cosine similarities between the query code snippet and all error type embeddings.
    code_norm = code_embedding / code_embedding.norm(p=2)
    error_type_norms = F.normalize(error_type_embeddings, p=2, dim=1)
    similarities = torch.matmul(error_type_norms, code_norm)

    # Exclude the query code snippet from recommendations.
    local_index = node_index
    similarities[local_index] = -float("inf")

    # Get the top-10 recommended error types.
    topk = 10
    top_sim_values, top_indices = torch.topk(similarities, topk)

    recommended_error_types = []
    for local_idx in top_indices.tolist():
        global_node_idx = local_idx + num_code_snippets  # convert local index to global index
        rec_error_type_id = inv_error_type_map.get(global_node_idx, None)
        if rec_error_type_id is not None:
            recommended_error_types.append(rec_error_type_id)
        else:
            recommended_error_types.append("Unknown Error Type")

    print(f"\nTop 10 recommended error types similar to '{query_name}':")
    for i, name in enumerate(recommended_error_types, 1):
        print(f"{i}. {name}")
    print("\n")
