import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torchvision


def show_random_samples(images, num_of_samples=6):
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
    # Let's look at the class to which it belongs
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
    print(f"NUMBER OF {title} IMAGES:")
    print(f" - normal: {normal}")
    print(f" - bacteria: {bacteria}")
    print(f" - virus: {virus}")
    print(f" total: {total}")

def summary_of_images(labels, np_images):
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
    loader = torch.utils.data.DataLoader( dataset, batch_size=6, shuffle=True)
    batch = next(iter(loader))
    x_images, y_labels = batch
    
    grid = torchvision.utils.make_grid((x_images / 2 + 0.5), nrow=3)
    plt.figure(figsize=(11, 11))
    plt.imshow(np.transpose(grid, (1,2,0)))
    print('labels: ', y_labels)