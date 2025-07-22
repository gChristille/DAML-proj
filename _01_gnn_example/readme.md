Data from Stanford Ribonanza RNA Folding competition (kaggle)

This part is based on https://www.kaggle.com/code/fnands/a-quick-gnn-baseline/notebook#Training

Despite changing the GNN and varying the hyperparameters (including the k for the k-NN),
I couldn't get the model to move away from a constant prediction.
I also tried bayesian optimization before realizing that the model was not learning at all.

Data is very noisy. Could achieve some results by using feature extraction from RNA models