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
import random
import numpy as np
from PIL import Image
import torch as th
import torchvision
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive
from torchvision import datasets

from tqdm import tqdm

### Preprocess dataset 
# The dataset is preprocesses to be in the expected form accepted by 
# the until function "image_dataset_loader.py" which assumes images
# are named in the form "label_index.datatype".

data_dir = "/home/sszabados/datasets/c4_mnist/"
raw_dir = data_dir+"data"
processed_dir = data_dir
resource_link = [("http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip",
                 "0f9a947ff3d30e95cd685462cbf3b847")]


def gen_rot_mnist_c4_jpg(samples=600):
    """
        Create rotated MNIST dataset under C4 group action. This does not augment the dataset,
        rather each image is randomly rotated by one of {0,pi/2,pi,3pi/2} radians.
    """

    mnist_dataset = datasets.MNIST(train=True, root=raw_dir, download=True)

    for i in range(min(samples, len(mnist_dataset))):
        image, target = mnist_dataset[i]
        # Select random rotation angle
        k = th.randint(low=0, high=4, size=(1,), dtype=int)
        image = image.rotate(90*k, expand=False)
        image.save(os.path.join(data_dir, f"{target}_{i}.JPEG"))
        

def convert_jpy_npy():
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg'))]    
    num_files = len(files)

    train_images = th.empty(size=(num_files, 28, 28, 1))
    train_lables = th.empty(size=(num_files,))
    
    convert_tensor = transforms.ToTensor()

    i = 0
    for file in tqdm(files):
        image_path = os.path.join(data_dir, file)
        label = int(file.split('_')[0])
        train_lables[i] = label
        image = convert_tensor(Image.open(image_path)).unsqueeze(1).reshape(-1, 28, 28, 1)
        image = 2.*image - 1. # normalize data range to [-1,1]
        train_images[i] = image
        i += 1

    np.save(os.path.join(data_dir, "train_images.npy"), train_images.numpy())
    np.save(os.path.join(data_dir, "train_labels.npy"), train_lables.numpy())


def gen_rot_mnist_jpg(samples=600):
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
    train = th.from_numpy(np.loadtxt(train_filename))
    test = th.from_numpy(np.loadtxt(test_filename))
    train_data = train[:, :-1].reshape(-1, 28, 28)
    train_data = (train_data * 256).round().type(th.uint8)
    train_labels = train[:, -1].type(th.uint8)
    train_set = (train_data, train_labels)
    # we ignore the validation set
    test_data = test[:, :-1].reshape(-1, 28, 28)
    test_data = (test_data * 256).round().type(th.uint8)
    test_labels = test[:, -1].type(th.uint8)
    test_set = (test_data, test_labels)
    print("Finished downloading files.")

    print("Generating label_index.JPG image training data...")
    print("Creating "+str(min(samples, len(train_labels)))+" images...")
    for i in range(min(samples, len(train_labels))):
        image = Image.fromarray(train_data[i].numpy())
        target = train_labels[i]
        image.save(os.path.join(data_dir, f"{target}_{i}.JPEG"))
    print("Finished.")


def gen_rot_mnist_npy(samples=600):
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
    train = th.from_numpy(np.loadtxt(train_filename))
    test = th.from_numpy(np.loadtxt(test_filename))
    train_data = train[:, :-1].reshape(-1, 28, 28).unsqueeze(1)
    train_data = train_data.reshape(-1, 28, 28, 1)
    train_data = (train_data * 256).round().type(th.uint8)
    train_data = train_data[0:samples]
    train_labels = train[:, -1].type(th.uint8)
    train_labels = train_labels[0:samples]
    train_set = (train_data, train_labels)
    # we ignore the validation set
    test_data = test[:, :-1].reshape(-1, 28, 28).unsqueeze(1)
    test_data = test_data.reshape(-1, 28, 28, 1)
    test_data = (test_data * 256).round().type(th.uint8)
    test_data = test_data[0:samples]
    test_labels = test[:, -1].type(th.uint8)
    test_labels = test_labels[0:samples]
    test_set = (test_data, test_labels)
    print("Finished downloading files.")

    # DEBUG
    # grid_img = torchvision.utils.make_grid(train_data[0:5].reshape(5,1,28,28), nrow = 1, normalize = True)
    # torchvision.utils.save_image(grid_img, f'tmp_imgs/rot_mnist_sample.pdf')

    print("Saving npz files...")
    with open(os.path.join(processed_dir, "train_images.npy"), "wb") as f:
        np.save(f, train_data.numpy())
    with open(os.path.join(processed_dir, "train_labels.npy"), "wb") as f:
        np.save(f, train_labels.numpy())
    with open(os.path.join(processed_dir, "test_images.npy"), "wb") as f:
        np.save(f, test_data.numpy())
    with open(os.path.join(processed_dir, "test_labels.npy"), "wb") as f:
        np.save(f, test_labels.numpy())
    print("Finished.")


def sample_rot_mnist_pdf(num_samples=10):
    num_classes = 10
    class_images = {}
    # Load image dataset from data_dir 
    # Assuming 'dataset' is a 3D array with shape (height, width, channels)
    images = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    num_images = len(images)
    resolution = Image.open(data_dir+images[0]).size[0]
    print(str(num_images)+", "+str(np.array(Image.open(data_dir+images[0])).shape))
    sample_images = th.zeros((num_samples, num_classes, 3, resolution, resolution))

    i = 0
    for image in images:
        if i < num_samples*100:
            label = int(image.split('_')[0])
            if label not in class_images:
                class_images[label] = []
            class_images[label].append(image)
            i += 1
        else:
            break

    transform = transforms.Compose([transforms.PILToTensor()])
    for label, images in class_images.items():
        selected_images = random.sample(images, num_samples)
        for i in range(len(selected_images)):
            sample_images[i,label,:,:,:] = transform(Image.open(data_dir+selected_images[i]))
     
    sample_images = sample_images/256

    grid_img = torchvision.utils.make_grid(sample_images.reshape(num_samples*num_classes, 3, resolution, resolution), nrow=num_classes, normalize=True)
    torchvision.utils.save_image(grid_img, f'tmp_imgs/rot_mnist_sample.pdf')


def main():
    # gen_rot_mnist_jpg(samples=6000)
    # gen_rot_mnist_npy(samples=6000)
    # gen_rot_mnist_c4_jpg(samples=60000)
    # sample_rot_mnist_pdf(num_samples=3)
    convert_jpy_npy()



if __name__ == '__main__':
    main()