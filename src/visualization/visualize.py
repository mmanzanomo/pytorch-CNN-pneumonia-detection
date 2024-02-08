import os

import numpy as np
import matplotlib.pyplot as plt
import random

import PIL.Image as Image

import torch
import torch.nn as nn
import torchvision


def show_random_samples(images, num_of_samples=6):
    """
    Displays subplots with randomly selected images, along with their shape, minimum, and maximum color values.

    Args:
        images (List[np.ndarray]): List of images as NumPy arrays.
        num_of_samples (int, optional): Number of random samples to display. Default is 6.
    """
    random_samples = random.sample(range(0, len(images)), num_of_samples)

    for i in range(len(random_samples)):
        temp_image = images[random_samples[i]]
        plt.subplot(2, 3, i+1)
        plt.axis("off")
        # cmap="gray" to view in grayscale. (If not, take the color scale)
        plt.imshow(temp_image, cmap="gray")
        plt.subplots_adjust(wspace = 0.5)
        plt.show()
        print("Shape:{0}, min_color:{1}, max_color:{2}".format(temp_image.shape,
                                                temp_image.min(),
                                                temp_image.max()))


def show_sample_per_class(labels, images):
    """
    Display a sample image for each unique class in the dataset.

    Args:
        labels (List[str]): List of labels corresponding to the images.
        images (List[np.ndarray]): List of images as NumPy arrays.
    """
    unique_labels = set(labels)
    plt.figure(figsize=(6,6))
    i=1
    for label in unique_labels:
        temp_image = images[labels.index(label)]
        plt.subplot(1,2, i)
        plt.axis("off")
        plt.title("{0} Class ({1})".format(label, labels.count(label)))
        i += 1
        plt.imshow(temp_image, cmap="gray")
    plt.show()


def count_labels(labels):
    """
    Count the occurrences of different labels in a given list.

    Args:
        labels (List[str]): List of labels.

    Returns:
        Tuple[int, int, int, int]: Tuple containing counts for 'NORMAL', 'PNEUMONIA_BACTERIA', 'PNEUMONIA_VIRUS', and total labels.

    Example:
        normal_count, bacteria_count, virus_count, total_count = count_labels(my_labels)
    """
    normal = 0
    bacteria = 0
    virus = 0
    total = 0

    for label in labels:
        if label == 'NORMAL':
            normal += 1
        elif label == 'PNEUMONIA_BACTERIA':
            bacteria += 1
        elif label == 'PNEUMONIA_VIRUS':
            virus += 1
        total += 1

    return normal, bacteria, virus, total


def print_three_labels_result(title, normal, bacteria, virus, total):
    """
    Print the number of images for three categories: 'normal', 'bacteria', and 'virus'.

    Args:
        title (str): Title for the result.
        normal (int): Number of normal images.
        bacteria (int): Number of bacteria images.
        virus (int): Number of virus images.
        total (int): Total number of images.

    Example:
        print_three_labels_result("Training", 500, 800, 700, 2000)
    """
    print(f"NUMBER OF {title} IMAGES:")
    print(f" - normal: {normal}")
    print(f" - bacteria: {bacteria}")
    print(f" - virus: {virus}")
    print(f" total: {total}")


def summary_of_images(labels, np_images):
    """
    Display a summary of images for each unique label.

    Displays a set of subplots, each representing a unique label with an associated image.
    The title of each subplot includes the label and the count of occurrences in the dataset.

    Args:
        labels (List[str]): List of labels corresponding to the images.
        np_images (List[np.ndarray]): List of images as NumPy arrays.

    Example:
        summary_of_images(my_labels, my_images)
    """
    unique_labels = set(labels)
    plt.figure(figsize=(8,3))
    i=1
    for label in unique_labels:
        temp_image = np_images[labels.index(label)]
        plt.subplot(1,len(unique_labels), i)
        plt.axis("off")
        plt.title("{0} \n({1})".format(label, labels.count(label)))
        i += 1
        plt.imshow(temp_image, cmap="gray")
    plt.suptitle('Class (number of samples):')
    plt.show()


def show_images(dataset):
    """
    Display a grid of images from the given PyTorch dataset along with their labels.

    Args:
        dataset (torch.utils.data.Dataset): PyTorch dataset containing images and labels.

    Note:
        This function assumes that the dataset provides images and labels in the format expected by the DataLoader.

    Example:
        show_images(my_dataset)
    """
    loader = torch.utils.data.DataLoader( dataset, batch_size=6, shuffle=True)
    batch = next(iter(loader))
    x_images, y_labels = batch
    
    grid = torchvision.utils.make_grid((x_images / 2 + 0.5), nrow=3)
    plt.figure(figsize=(11, 11))
    plt.imshow(np.transpose(grid, (1,2,0)))
    print('labels: ', y_labels)


def plot_loss_history(history):
    """
    Plots the training and validation loss history.

    Parameters:
    - history: List or array containing loss history for training and validation.
    """
    history = np.array(history)

    # Plot training and validation loss
    plt.plot(history[:,:2])
    
    plt.legend(['Train Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    
    plt.show()


def plot_accuracy_history(history):
    """
    Plots the training and validation accuracy history.

    Parameters:
    - hist: List or array containing accuracy history for training and validation.
    """
    history = np.array(history)

    # Plot training and validation accuracy
    plt.plot(history[:,2:])
    
    plt.legend(['Train Acc', 'Val Acc'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    
    plt.show()


def classify_random_image(model, image_transform, test_path, classes, device='cpu'):
    """
    Classifies an image using a pre-trained PyTorch model.

    Args:
        model (torch.nn.Module): Pre-trained PyTorch model.
        image_transform (torchvision.transforms.Compose): Image transformation applied to the model input.
        test_path (str): Path to the test set with subdirectories for each class.
        classes (list): List of class names used for classification.
        device (str, optional): Device used for inference ('cuda' or 'cpu').

    Returns:
        None. Prints information about the image and prediction.

    Example:
        classify(model, transform, '/path/to/test/set/', ['NORMAL', 'PNEUMONIA_BACTERIA', 'PNEUMONIA_VIRUS'])
    """
    model.eval()

    # Get a random image
    label_dir = random.choice(classes)
    choice_path = test_path + label_dir + '/'
    sample = random.sample( os.listdir(choice_path), k=1)
    image = Image.open(choice_path + sample[0] )
    plt.imshow(image, cmap='gray')

    # Preprocessing
    image = image_transform(image).float()
    image = image.unsqueeze(0)

    # Charge to same device (cpu or gpu)
    image=image.to(device)
    output = model(image).to(device)

    # Prediction
    s = nn.Softmax(dim=1)
    soft = s(output).to('cpu').detach()
    soft = soft.numpy()
    _, pred = torch.max(output.data, 1)
    
    print(f'image: {sample[0]}\npred: {classes[pred.item()]}')
    print(f'-Softmax: \nNormal: {(soft[0][0]):.4f} | Virus: {(soft[0][1]):.4f} | Bacteria: {(soft[0][2]):.4f}')
