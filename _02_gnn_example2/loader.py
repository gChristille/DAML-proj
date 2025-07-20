from torch_geometric.data import Dataset, Data
import networkx as nx
import torch

class GraphDataset(Dataset):
    def __init__(self, graphs: list[nx.Graph], ys: list[torch.Tensor] = None):
        super().__init__()
        self.graphs = graphs
        self.xs = []
        self.edge_indices = []
        self.ys = ys

        for graph_idx, graph in enumerate(self.graphs):
            self.xs.append([])
            self.edge_indices.append(0)
            if not isinstance(graph, nx.Graph):
                raise TypeError("All elements in the dataset must be of type nx.Graph")
            for node_idx, node in enumerate(graph.nodes):
                node_features = graph.nodes.data()[node]["features"]
                self.xs[graph_idx].append(torch.tensor(node_features))

            self.edge_indices[graph_idx] = torch.tensor(list(graph.edges)).t().contiguous()

            self.xs[graph_idx] = torch.stack(self.xs[graph_idx], dim=0)

        self.ys = [torch.argmax(torch.tensor(y).reshape(-1, 1).T, dim=1) for y in ys]


    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return Data(x=self.xs[idx], edge_index=self.edge_indices[idx], y=self.ys[idx])

    @property
    def num_classes(self):
        return torch.max(torch.stack(self.ys)).item() + 1 if self.ys else 0
    
    @property
    def num_node_features(self):
        return self.xs[0].shape[1]