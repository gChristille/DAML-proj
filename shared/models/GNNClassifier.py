import torch_geometric.nn as nn
import torch
from torch.nn import Linear, Sequential, ModuleList, Identity, ReLU
import torch.nn.functional as F
from torch_geometric.data import Dataset


class GNNClassifier(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_classes: int,
        hidden_ch: int,
        message_passing_module: torch.nn.Module = nn.GraphConv,
        message_passing_layers: int = 3,
        fc_layers: int = 1,
        fc_hidden_dim: int = None,
        dropout: float = 0,
        seed: int = 123,
    ) -> None:
        
        if fc_hidden_dim is None:
            fc_hidden_dim = hidden_ch

        super(GNNClassifier, self).__init__()
        torch.manual_seed(seed)

        self.convolutions = ModuleList()
        self.convolutions.append(
            message_passing_module(num_node_features, hidden_ch)
        )

        if message_passing_layers > 1:
            for _ in range(message_passing_layers - 1):
                self.convolutions.append(message_passing_module(hidden_ch, hidden_ch))

        self.dropout_prob = dropout


        if fc_layers == 0:
            self.mlp = Identity()

        elif fc_layers == 1:
            self.mlp = Sequential(
                Linear(self.convolutions[-1].out_channels, num_classes),
            )
        elif fc_layers >= 2:
            first_layer = Sequential(
                Linear(self.convolutions[-1].out_channels, fc_hidden_dim),
                ReLU(),
            )

            if fc_layers == 2:
                middle_layers = []
            else:
                middle_layers = []
                for _ in range(fc_layers - 2):
                    middle_layers.extend([Linear(fc_hidden_dim, fc_hidden_dim), ReLU()])

            last_layer = Linear(fc_hidden_dim, num_classes)

            layers = [first_layer] + middle_layers + [last_layer]
            self.mlp = Sequential(*layers)

        # self.lin_layers = ModuleList()
        # self.lin_layers.append(
        #     Linear(self.convolutions[-1].out_channels, fc_hidden_dim)
        # )
        # if fc_layers > 1:
        #     for _ in range(fc_layers - 1):
        #         self.lin_layers.append(Linear(fc_hidden_dim, fc_hidden_dim))

    def forward(self, x, edge_index, batch=None, weights=None) -> torch.Tensor:
        for conv in self.convolutions:
            x = conv(x, edge_index, edge_weight=weights)
            x = x.relu()
            x = F.dropout(x, p=self.dropout_prob, training=self.training)

        # Use PyTorch Geometric's global_max_pool which handles batching correctly
        if batch is None:
            # If no batch info, assume single graph
            x = torch.max(x, dim=0, keepdim=True).values
        else:
            # Proper batched global pooling
            x = nn.global_max_pool(x, batch)

        x = self.mlp(x)

        return x


    def get_model_inputs_from_batch(self, batch):
        """
        Extracts the necessary inputs from a batch for the forward pass.
        """
        x = batch.x
        edge_index = batch.edge_index
        batch_info = batch.batch

        return x, edge_index, batch_info
    
    def get_labels_from_batch(self, batch):
        """
        Extracts the labels from a batch.
        """
        return batch.y









# from torch_geometric.nn import global_max_pool

# class GNNClassifier(torch.nn.Module):
#     def __init__(self, dataset, hidden_ch, dropout=0):
#         super(GNNClassifier, self).__init__()
#         torch.manual_seed(1234)
#         self.conv1 = nn.GraphConv(dataset.num_node_features, hidden_ch)
#         self.conv2 = nn.GraphConv(hidden_ch, hidden_ch)
#         self.conv3 = nn.GraphConv(hidden_ch, hidden_ch)
#         self.dropout_prob = dropout
        
#         self.mlp = Sequential(
#             Linear(self.conv2.out_channels, dataset.num_classes),
#         )

#     def forward(self, x, edge_index, batch=None):
#         x = self.conv1(x, edge_index)
#         x = x.relu()
#         x = self.conv2(x, edge_index)
#         x = x.relu()
#         x = self.conv3(x, edge_index)
        
#         # Use PyTorch Geometric's global_max_pool which handles batching correctly
#         if batch is None:
#             # If no batch info, assume single graph
#             x = torch.max(x, dim=0, keepdim=True).values
#         else:
#             # Proper batched global pooling
#             x = global_max_pool(x, batch)
        
#         x = self.mlp(x)
#         return x
