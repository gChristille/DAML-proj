import torch
import math


def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return correct / total


def train_test_split(dataset, sizes=(0.7, 0.15)):
    """
    Splits the dataset into training and testing sets.
    
    Args:
        dataset: The dataset to split.
        train_size: Proportion of the dataset to include in the training set.
    
    Returns:
        A tuple containing the training and testing datasets.
    """

    assert len(sizes) == 2, "Sizes must be a tuple of two elements (train_size, val_size)"

    train_size, val_size = sizes
    total_size = len(dataset)
    
    train_end = int(total_size * train_size)
    val_end = int(total_size * (train_size + val_size))

    train_dataset, val_dataset, test_dataset = dataset[:train_end], dataset[train_end:val_end], dataset[val_end:]

    return train_dataset, val_dataset, test_dataset


def closest_factors(n):
    """
    Utility function to find the closest factors of a number n.
    Useful for determining the best shape for a grid of subplots.
    """
    sqrt_n = int(math.isqrt(n))  # Start from sqrt(n) and go down
    for i in range(sqrt_n, 0, -1):
        if n % i == 0:
            return i, n // i  # i * (n // i) == n, and they are closest
