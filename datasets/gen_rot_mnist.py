"""
    Author: Spencer Szabados
    Date: 2024-01-03

    File contains scripts for generating the rotmnist dataset proposed in 
    (http://www.dumitru.ca/files/publications/icml_07.pdf) and used in 
    (https://proceedings.mlr.press/v162/birrell22a/birrell22a.pdf) as a
    benchmark for evaluating equivariance conditioning of generative 
    models.

    Implementaion is based on:
    (https://github.com/david-knigge/separable-group-convolutional-networks/blob/main/datasets/mnist_rot.py)

    This file only generates a dataset that contains 12,000 images, so all 1% figures reported in the paper
    are computed based on the number of samples requred for 1% of 60,000 directly in order to keep comparison
    fair throught.
"""


import os
import numpy as np
from PIL import Image

import torchvision
import torch as th
from torchvision.datasets.utils import download_and_extract_archive


### Preprocess dataset 
# The dataset is preprocesses to be in the expected form accepted by 
# the until function "image_dataset_loader.py" which assumes images
# are named in the form "label_index.datatype".

data_dir = "/home/sszabados/datasets/rot_mnist_3000/"
raw_dir = data_dir+"data"
processed_dir = data_dir
resource_link = [("http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip",
                 "0f9a947ff3d30e95cd685462cbf3b847")]


def gen_rot_mnist(samples=600):
    """Download the MNIST data if it doesn't exist in processed_folder already."""
    
    os.makedirs(raw_dir, exist_ok=True)
    # download files
    print("Downloading files...")
    for url, md5 in resource_link:
        filename = url.rpartition("/")[2]
        download_and_extract_archive(
            url, download_root=raw_dir, filename=filename, md5=md5
        )
    # process and save as torch files
    print("Processing...")
    train_filename = os.path.join(
        raw_dir, "mnist_all_rotation_normalized_float_train_valid.amat"
    )
    test_filename = os.path.join(
        raw_dir, "mnist_all_rotation_normalized_float_test.amat"
    )
    train_val = th.from_numpy(np.loadtxt(train_filename))
    test = th.from_numpy(np.loadtxt(test_filename))
    train_val_data = train_val[:, :-1].reshape(-1, 28, 28)
    train_val_data = (train_val_data * 256).round().type(th.uint8)
    train_val_labels = train_val[:, -1].type(th.uint8)
    training_set = (train_val_data, train_val_labels)
    # we ignore the validation set
    test_data = test[:, :-1].reshape(-1, 28, 28)
    test_data = (test_data * 256).round().type(th.uint8)
    test_labels = test[:, -1].type(th.uint8)
    test_set = (test_data, test_labels)
    print("Finished downloading files.")

    print("Saving .pt files")
    with open(os.path.join(processed_dir, "training.pt"), "wb") as f:
        th.save(training_set, f)
    with open(os.path.join(processed_dir, "test.pt"), "wb") as f:
        th.save(test_set, f)

    print("Generating label_index.JPG image training data...")
    print("Creating "+str(min(samples, len(train_val_labels)))+" many images...")
    for i in range(min(samples, len(train_val_labels))):
        image = Image.fromarray(train_val_data[i].numpy())
        target = train_val_labels[i]
        image.save(os.path.join(data_dir, f"{target}_{i}.JPEG"))
    print("Finished.")
       

def main():
    gen_rot_mnist(samples=3000)


if __name__ == '__main__':
    main()