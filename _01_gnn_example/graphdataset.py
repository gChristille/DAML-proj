import torch
import numpy as np
import polars as pl
from torch_geometric.data import Dataset, Data
from sklearn.preprocessing import OneHotEncoder
import re

def nearest_adjacency(sequence_length, n=2, loops=True):
    base = np.arange(sequence_length)
    connections = []
    for i in range(-n, n + 1):
        if i == 0 and not loops:
            continue
        elif i == 0 and loops:
            stack = np.vstack([base, base])
            connections.append(stack)
            continue

        neighbours = base.take(range(i,sequence_length+i), mode='wrap')
        stack = np.vstack([base, neighbours])
        
        if i < 0:
            connections.append(stack[:, -i:])
        elif i > 0:
            connections.append(stack[:, :-i])

    return np.hstack(connections)

class GraphDataset(Dataset):
    def __init__(self, 
                 parquet_name, 
                 edge_distance=1, 
                 allow_loops=False,
                 root=None, 
                 transform=None, 
                 pre_transform=None, 
                 pre_filter=None):
        
        super().__init__(root, transform, pre_transform, pre_filter)
        # Set csv name
        self.parquet_name = parquet_name
        # Set edge distance
        self.edge_distance = edge_distance
        self.allow_loops = allow_loops

        
        self.df = pl.read_parquet(self.parquet_name)
        self.df = self.df.drop(pl.col(colname) for colname in self.df.columns if colname.startswith("reactivity_error"))
        self.df = self.df.filter(pl.col("SN_filter") == 1.0)
        self.df = self.df.head(20_000)

        self.init_sequences()
        # print("Initialized sequences with shape:", self.seq_vector.shape)
        self.init_reactivities()
        # print("Initialized reactivities with shape:", self.reactivities.shape)
        self.init_edges()
        # print("Initialized edges with lengths:", list(self.edges.keys()))


    def init_sequences(self):
        seq_df = self.df.select("sequence")
        seq_vector = seq_df.to_numpy().squeeze().reshape(-1, 1)  # Reshape to ensure it's a 2D array

        lengths = [len(seq[0]) for seq in seq_vector]
        self.lengths = np.array(lengths)

        node_encoder = OneHotEncoder(sparse_output=False, max_categories=5)

        encoding_dict = np.array(['A', 'G', 'U', 'C']).reshape(-1, 1)

        node_encoder = node_encoder.fit(encoding_dict)

        cum_lengths = np.cumsum(lengths)

        new_seq_vector = np.zeros((len(encoding_dict), cum_lengths[-1]), dtype=int)
        for i, seq in enumerate(seq_vector):
            start = cum_lengths[i - 1] if i > 0 else 0  
            end = cum_lengths[i]
            new_seq_vector[:, start:end] = node_encoder.transform(np.array(list(seq[0])).reshape(-1, 1)).T

        self.cum_lengths = cum_lengths
        self.seq_vector = torch.tensor(new_seq_vector, dtype=torch.float32).T


    def init_reactivities(self):
        reactivity_match = re.compile('(reactivity_[0-9])')
        reactivity_names = [col for col in self.df.columns if reactivity_match.match(col)]

        reactivity_df = self.df.select(reactivity_names)

        mat = reactivity_df.to_numpy()

        reactivities = np.zeros(self.cum_lengths[-1])
        for i, row in enumerate(mat):
            row = row[~np.isnan(row)]
            start = self.cum_lengths[i - 1] if i > 0 else 0
            end = self.cum_lengths[i]

            # Ensure the row fits within the cumulative lengths
            reactivities[start:end] = np.pad(row[:(end - start)], (0, end - start - len(row)), mode='constant', constant_values=0)

        self.reactivities = torch.Tensor(reactivities)


    def init_edges(self):
        edges = {}

        unique_lengths = np.unique(self.lengths)
        # print(f"Detected {len(unique_lengths)} unique lengths. Computing edges for each.")
        for length in unique_lengths:
            if length not in edges:
                edges[length] = nearest_adjacency(length, n=self.edge_distance, loops=self.allow_loops)
                edges[length] = torch.tensor(edges[length], dtype=torch.long)

        self.edges = edges

    def len(self):
        return len(self.df)

    def get(self, idx):
        start = self.cum_lengths[idx - 1] if idx > 0 else 0
        end = self.cum_lengths[idx]

        sequence_vector = self.seq_vector[start:end, :]
        reactivity_vector = self.reactivities[start:end]
        edges = self.edges[sequence_vector.shape[0]]

        return Data(x=sequence_vector, edge_index=edges, y=reactivity_vector)