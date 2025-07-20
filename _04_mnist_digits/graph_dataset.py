from tqdm import tqdm
import torch
from torch_geometric.data import Dataset, Data
from shared.img_to_graph import img_to_graph

class GraphDataset(Dataset):
    def __init__(self, images, labels, use_weighted_edges=True):
        super().__init__()
        print(f"GraphDataset.__init__ called with {len(images)} images")
        self.images = images
        self.labels = labels

        Xs = [0] * len(images)
        edge_indices = [0] * len(images)
        edge_weights = [0] * len(images)
        self.centroids = [0] * len(images)
        self.pixel_counts = [0] * len(images)


        # start_time = time.time()
        # last_checkpoint = 0
        
        for i, img in tqdm(enumerate(images), position=0, leave=True, total=len(images)):
            X, edge_index, edge_weight, centroid, pixel_count = img_to_graph(img, return_pixel_counts=True)
            Xs[i] = X
            edge_indices[i] = edge_index
            self.centroids[i] = centroid
            self.pixel_counts[i] = pixel_count

            if use_weighted_edges:
                edge_weights[i] = edge_weight
            else:
                edge_weights[i] = torch.ones(edge_index.size(1), dtype=torch.float)

            # if i % 1000 == 0 and i > 0:
            #     elapsed_time = time.time() - start_time
            #     images_processed = i - last_checkpoint
            #     print(f"Processed {images_processed} images (up to {i}) in {elapsed_time:.2f} seconds", flush=True)
            #     start_time = time.time()  # Reset start time for next batch
            #     last_checkpoint = i

        self.Xs = Xs
        self.edge_indices = edge_indices
        self.edge_weights = edge_weights
        print("GraphDataset initialization complete!")

    def len(self):
        return len(self.images)

    def get(self, idx):
        return Data(x=self.Xs[idx], # (Nnodes, Nfeatures)
                    edge_index=self.edge_indices[idx].T, # (2, Nedges)
                    edge_weight=self.edge_weights[idx],
                    y=self.labels[idx])