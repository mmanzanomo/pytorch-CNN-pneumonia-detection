import time

import numpy as np

import torch
import torch.nn as nn

from utils import check_device

torch.manual_seed(585)
np.random.seed(585)


def save_checkpoint(model, epoch, optimizer, best_acc):
    """
    Save a checkpoint of the model's state.

    Parameters:
        model (torch.nn.Module): The PyTorch model to be saved.
        epoch (int): The current training epoch.
        optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        best_acc (float): The best accuracy achieved during training.

    Returns:
        None

    The saved checkpoint is stored in the '../models/model_checkpoint.pth.tar' file.
    """
    state = {
        'epoch': epoch+1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'accuracy': best_acc,
    }
    torch.save( state, '../models/model_checkpoint.pth.tar')


def train_loop(dataloader, model, criterion, optimizer):
    """
    Perform a single training loop on the provided model.

    The function trains the model on the provided DataLoader for one epoch.
    It computes predictions, calculates the loss, and performs backpropagation to update model parameters.
    The training progress is printed every 32 batches.

    Parameters:
        dataloader (torch.utils.data.DataLoader): The DataLoader providing the batches of training data.
        model (torch.nn.Module): The PyTorch model to be trained.
        criterion (torch.nn.Module): The loss criterion used for training.
        optimizer (torch.optim.Optimizer): The optimizer used for updating model parameters.

    Returns:
    - t_loss: List of training losses for each batch.
    - t_acc: Training accuracy as a percentage.
    """
    model.train()
    device = check_device.set_device()
    size = len(dataloader.dataset)
    
    t_loss = []
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        X=X.to(device)
        y=y.to(device)
        
        # Compute prediction and loss
        pred = model(X).to(device)          
        loss = criterion(pred, y)
        t_loss.append(loss.item())
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch) % 32 == 0:
            loss, current = loss.item(), batch * len(y)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    correct /= size
    t_acc = (100*correct)
    
    return t_loss, t_acc


def test_loop(dataloader, model, criterion):
    """
    Evaluate the model on the provided test data.

    The function evaluates the model on the provided DataLoader and returns the accuracy,
    average test loss, true labels, and predicted labels.

    Parameters:
        dataloader (torch.utils.data.DataLoader): DataLoader providing the batches of test data.
        model (torch.nn.Module): The PyTorch model to be evaluated.
        criterion (torch.nn.Module): The loss criterion used for evaluation.

    Returns:
    - accuracy: Test accuracy as a percentage.
    - test_loss: Average test loss.
    - true_labels: List of true labels.
    - pred_labels: List of predicted labels.
    """
    model.eval()
    device = check_device.set_device()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X=X.to(device)
            y=y.to(device)

            pred = model(X).to(device)
            test_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            true_labels.append(y.detach().cpu().numpy().tolist())
            pred_labels.append( (np.argmax(pred.detach().cpu().numpy(), axis=1)).tolist())
            
    # Flattens the list of lists into one.
    flat_true_list = []
    for item in true_labels:
        flat_true_list += item
    
    flat_pred_list = []
    for item in pred_labels:
        flat_pred_list += item
    
    test_loss /= num_batches
    correct /= size
    accuracy = (100*correct)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return accuracy, test_loss, flat_true_list, flat_pred_list


def train_model(model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, num_epochs=5):
    """
    Train the model using the provided training and validation data.

    The function trains the model for the specified number of epochs, evaluating on the validation
    data after each epoch. It also implements early stopping if the validation accuracy does not improve.

    Parameters:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        valid_dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        criterion (torch.nn.Module): The loss criterion used for training and evaluation.
        optimizer (torch.optim.Optimizer): The optimizer used for updating model parameters.
        num_epochs (int, optional): Number of training epochs. Default is 5.

    Returns:
    - history: List of lists containing training history for each epoch.
    """
    best_acc = 0
    since = time.time()
    early_stop = 10
    stop = 0
    history = []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} / {num_epochs} \n-------------------------------")
        train_loss, train_acc = train_loop(train_dataloader, model, criterion, optimizer)
        test_acc, test_loss, _, _ = test_loop(valid_dataloader, model, criterion)
        
        scheduler.step(test_loss)
        history.append([ np.mean(train_loss), test_loss, train_acc, test_acc])
        
        if(test_acc > best_acc):
            best_acc = test_acc
            save_checkpoint(model, epoch, optimizer, best_acc)
            stop = 0
        elif stop == early_stop:
            print("Early stopping\n")
            break
        else:
            stop += 1

    time_elapsed = time.time() - since
    print("Done!")
    print(f" -Complete in {(time_elapsed//60):.0f}m {(time_elapsed%60):.0f}s")
    print(f"  Best accuracy: {best_acc:.1f}%")
    
    return history


def test_model(test_dataloader, model, criterion):
    """
    Evaluate the trained model on the provided test data.

    The function evaluates the trained model on the provided DataLoader for test data
    and returns the accuracy, average test loss, true labels, and predicted labels.

    Parameters:
        test_dataloader (torch.utils.data.DataLoader): DataLoader for test data.
        model (torch.nn.Module): The PyTorch model to be evaluated.
        criterion (torch.nn.Module): The loss criterion used for evaluation.

    Returns:
    - accuracy: Test accuracy as a percentage.
    - test_loss: Average test loss.
    - true_labels: List of true labels.
    - pred_labels: List of predicted labels.
    """
    print("Test evaluation:")
    accuracy, test_loss, true_labels, pred_labels = test_loop(test_dataloader, model, criterion)
    print("Done!")
    return accuracy, test_loss, true_labels, pred_labels


class CNN_NeuralNetwork(nn.Module):
    """
    Convolutional Neural Network (CNN) architecture for image classification.

    Input:
        - x (torch.Tensor): Input tensor with shape (batch_size, in_channels, height, width).

    Output:
        - logits (torch.Tensor): Output tensor with shape (batch_size, 3), representing class logits.
    """
    def __init__(self, in_channels=1, out_channels_1=32, out_channels_2=64):
        super(CNN_NeuralNetwork, self).__init__()
        self.conv2d_stack = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels_1,
                      kernel_size=3,
                      padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(out_channels_1),
            nn.Conv2d(in_channels=out_channels_1,
                      out_channels=out_channels_2,
                      kernel_size=3,
                      padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(out_channels_2),
            nn.Conv2d(in_channels=out_channels_2,
                      out_channels=out_channels_2*2,
                      kernel_size=3,
                      padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(out_channels_2*2),
        )
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear((out_channels_2*2)*32*32, 128), 
            nn.ReLU(),
            #nn.Dropout(p=0.5, inplace=False),
            #nn.Linear(256, 128),
            #nn.ReLU(),
            #nn.Dropout(p=0.5, inplace=False),
            nn.Linear(128, 94),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(94, 62),
            nn.ReLU(),
            #nn.Dropout(p=0.5, inplace=False),
            nn.Linear(62, 21),  
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(21, 3),  
        )

    def forward(self, x):
        x = self.conv2d_stack(x)
        x = self.flatten(x)                          
        logits = self.linear_stack(x)
        return logits
