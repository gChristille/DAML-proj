# Project overview

This project is part of the final exam of Gioele Christille for the course "Data Analysis in experimental physics with Machine Learning" taught by Professor Paolo Meridiani at University of Turin. 

Goal: The project aims at exploring the viability of GNNs in image recognition by taking advantage of image segmentation algorithms to extract superpixels that work as nodes in the graphs. The project is comprised of many folders but only the ones numbered 03-05 will be presented in the final presentation. Folders 01 and 02 were mainly used to understand how GNNs work, but they do contain some optimizations (bayesian opt for hyperparams and message passing module comparisons) that haven't been used anywhere else.

To run folders 00 and 03-05, download from [here](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) the dataset and run the notebooks in order.

To run folder 01, follow the link in its readme and download the dataset.

To run the last notebook in folder 02, download the data at the url in its readme (the first two notebooks in folder 02 use TUDataset from torch.)

# Writeup of what the project includes

## Folder 00
This is the project's state at the time of the group presentation. The "modules" folder includes utilities divided by context. The notebooks:

- 0b: Visualize histological data
- 1a: Train resnet18 on CT using multiclass classification
- 1b: Train resnet18 on histological images using multiclass
- 2a: Train resnet18 on CT binary classification (healthy/sick)
- Lung: Train NNs on Lung tabular data

## Folder 01
This folder was used to familiarize myself with GNNs and understand how data should be passed to them. It uses dataset from a Kaggle competition about RNA sequences and uses k-NN to create a graph from a sequence of bases. The notebooks:

- 00: shows how data is stored
- 01: trains an EdgeCNN for a single iteration of hyperparameters
- 02: uses bayesian optimization to optimize NN architecture and hyperparameters (both of the learning process and of the graph construction)
- 03: performs model evaluation on the best models found. It also shows that even the best models just predict a constant value and still manage to achieve around .25 clipped MAE (compared to the competition winner of .13), since the dataset is looks very noisy.

## Folder 02
Uses other datasets that are notoriously easier to manage for GNNs. The notebooks:

- 01: trains and tests a GNN using GCNConv on a TUDataset
- 02: trains and tests a GNN using GraphConv (better) on a TUDataset. From this point on, the default message-passing module for GNNClassifier has been set to GraphConv
- 03: uses a new graph dataset that is made in a way such that node or edge attributes aren't enough alone and only when both are combined can good performance be achieved. The construction of this dataset is detailed in the link in the folder readme, but the summary is that the graphs are created by scattering points in clusters of different shapes in 3 possible combinations. Points are assigned a color based on the cluster they belong to, and connected using k-NN. The graph is then mapped to another manifold where clusters can't be seen by eye. The role of the GNN is to determine in which of the 3 possible combinations the graph was orinally created.

## Folder 03

This folder includes the usage of resnet for CT image classification to have a CNN baseline.Partly based on folder 00. The notebooks:

- 00: visualize dataset
- 01: use resnet to extract features that are then saved as tensors
- 02: train MLP and see that it overfits
- 03: use PCA to reduce the dimensionality from 960

## Folder 04

Tries to use a GNN on the mnist digits
The notebooks:

- 00: visualize digits, segment images (using quickshift) and compute RAGs (using skimage)
- 01: create the dataset and save it
- 01b: use voronoi reconstruction to get back an image that's similar to the original. The better the voronoi image looks, the more information the GNN has available.
- 02: training the GNN using only 1 feature per node (pixel color)
- 03: training the GNN by also using the centroid of the segmented region as features (total: 3 features: color, centerX, centerY)
- 04: count how many params an EdgeConv layer has to make sure it uses an MLP with a total of 3 layers of dimensions [256, 128, 128]
- test: make sure the graphs are being made correctly

## Folder 05
The main goal of the project, since it uses a real dataset that's not as easy as the MNIST digits. The notebooks:

- 00: augments and saves the dataset. This results in a 10-fold increase in the number of images, but only 1K images will be used since making a graph from an image is expensive (1 second means 3-4 hours for the full dataset).
- 01: shows how the graphs are made from a sample image.
- 02: make a dataset if it's not saved already and train the model. Training takes around 1h on my CPU and maxes all my 12 cores (probably for loading data only)


## Folder "shared"
Includes the utility functions used throughout the project. The files:

- img_to_graph: makes a graph (RAG) from an image
- training: includes the function used to train the classifier
- utils: includes utilities that probably aren't used anywhere (except for accuracy)
- models/GNNClassifier: contains the main model. I have made it very modular to always use this one class. It's made of a sequential set of messsage-passing-layers followed by max pooling and an MLP. Any part can be changed without having to rewrite the whole class. The function "shared.training.train_classifier" expects the classifier class to have the methods :
`get_model_inputs_from_batch(self, batch), 
get_labels_from_batch(self, batch)` that have a default but can be overwritten in the notebooks to train models on different datasets. This allows the same model class to be used on both weighted and unweighted graphs (note: I couldn't find a way to not have weighted graphs return nans, so all the results are based on unweighted graphs. Still, there are places where the weights are displayed to show their values). Also, 05/02 shows how the model class can be subclassed to use the same MLP part but a different message-passing part.



