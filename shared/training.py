
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.optim import Optimizer
from torch.nn import Module
import torch
from shared.utils import calculate_accuracy

def train_classifier(
    N_epochs: int = 100,
    model: Module = None,
    train_dataloader: TorchDataLoader | PyGDataLoader = None,
    val_dataloader: TorchDataLoader | PyGDataLoader = None,
    optimizer: Optimizer = None,
    loss: Module = None,
    output_loc="stdout",
) -> tuple[list[float], list[float]]:
    overall_train_acc = []
    overall_val_acc = []
            #"250/250   0.314   0.314  0.314   0.314\n"
    header = "Epoch    T Loss  T Acc  V Loss  V Acc\n"
    
    if output_loc != "stdout" and output_loc is not None:
        writing_to_file = True
        f = open(output_loc, "w+")
        f.write(header)
    elif output_loc == "stdout":
        writing_to_file = False
        print(header)
    else:
        writing_to_file = False

    for epoch in range(N_epochs):  # Reduced epochs for debugging
        model.train()
        train_losses = []
        train_accuracies = []

        for batch in train_dataloader:
            optimizer.zero_grad()
            out = model(*model.get_model_inputs_from_batch(batch))
            loss_value = loss(out, model.get_labels_from_batch(batch))

            # Calculate accuracy
            acc = calculate_accuracy(out, model.get_labels_from_batch(batch))
            train_accuracies.append(acc)

            train_losses.append(loss_value.item())
            loss_value.backward()

            optimizer.step()

        overall_train_acc.append(torch.mean(torch.tensor(train_accuracies)).item())

        # Validation
        model.eval()
        val_losses = []
        val_accuracies = []
        with torch.no_grad():
            for batch in val_dataloader:
                out = model(*model.get_model_inputs_from_batch(batch))
                loss_value = loss(out, model.get_labels_from_batch(batch))
                acc = calculate_accuracy(out, model.get_labels_from_batch(batch))
                val_losses.append(loss_value.item())
                val_accuracies.append(acc)

        overall_val_acc.append(sum(val_accuracies) / len(val_accuracies))

        epoch_num = f"{epoch + 1}/{N_epochs}" 
        epoch_output = (
            f"{epoch_num:<7}  "  # epoch
            f"{torch.mean(torch.tensor(train_losses)).item():<5.3f}   "  # train loss
            f"{overall_train_acc[-1]:<5.3f}   "  # train accuracy
            f"{torch.mean(torch.tensor(val_losses)).item():<5.3f}   "  # val loss
            f"{overall_val_acc[-1]:<5.3f}"
        )  # val accuracy



        if writing_to_file:
            f.write(epoch_output + "\n")

        elif output_loc == "stdout":
            print(epoch_output)


    f.close() if writing_to_file else None

    return overall_train_acc, overall_val_acc