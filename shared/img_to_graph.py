from skimage import segmentation as segm
import numpy as np
import skimage.graph as g
import networkx as nx
from pprint import pprint
from matplotlib.collections import LineCollection
from matplotlib import colormaps as cm
import torch

def find_centroid(segmented_image):
    """Find the centroid of each segment in the segmented image."""
    labels = np.unique(segmented_image)
    centroids = {}
    for label in labels:
        coords = np.column_stack(np.where(segmented_image == label))
        centroids[label] = coords.mean(axis=0)
    return centroids



def img_to_graph(img, quickshift_params=None, rag_params=None, return_pixel_counts=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if type(img) is not torch.Tensor:
        img = torch.tensor(img)

    if img.ndim == 2:
        img = img.unsqueeze(-1).repeat((1,1,3))
    elif img.ndim == 3 and img.shape[2] == 1:
        img = img.repeat(1, 1, 3)

    if quickshift_params is None:
        quickshift_params = {"kernel_size": 3, "max_dist": 4, "ratio": 0.4}
    
    if rag_params is None:
        rag_params = {"mode": "similarity"}

    segmented_image = segm.quickshift(img.numpy(), **quickshift_params)
    graph: g.RAG = g.rag_mean_color(img.numpy(), segmented_image, **rag_params)

    edge_index = torch.tensor(list(graph.edges)).t().contiguous() # (2, num_edges)
    X = torch.zeros(len(graph.nodes)) # (mean color of each node)
    centroids = torch.zeros((len(graph.nodes), 2)) # (x,y coordinates)
    pixel_counts = torch.zeros(len(graph.nodes)) # (number of pixels in each segment)

    centroids_dict = find_centroid(segmented_image)
    
    for node_idx, node in enumerate(graph.nodes):
        node_attr = graph.nodes[node]["mean color"]
        X[node_idx] = torch.mean(torch.tensor(node_attr))
        centroids[node_idx, :] = torch.tensor(centroids_dict[node])
        pixel_counts[node_idx] = float(graph.nodes[node]["pixel count"])

    edge_weights = torch.zeros(edge_index.shape[1])  # (num_edges,)
    edges = list(graph.edges)
    for i in range(len(edges)):
        n1, n2 = edges[i]
        edge_weights[i] = torch.sqrt(
            torch.sum((centroids[n1] - centroids[n2]) ** 2))

    # Normalizations
    edge_weights = edge_weights / (torch.sqrt(torch.tensor(2.))*img.shape[0])  # Normalize weights to [0, 1]
    X = X / 255  # Normalize colors to [0, 1]
    edge_weights = -torch.log(edge_weights)
    edge_weights /= edge_weights.max()  # Normalize edge weights to [0, 1]

    if not return_pixel_counts:
        return X, edge_index.T, edge_weights, centroids
    else:
        return X, edge_index.T, edge_weights, centroids, pixel_counts