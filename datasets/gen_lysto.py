"""
    Author: Spencer Szabados
    Date: 2023-12-24

    File contains scripts for generating LYSTO dataset (https://zenodo.org/records/3513571) 
    which contains various (native resolution 299x299 px) images of human cancer cells.
"""
import os
import re
import h5py
import numpy as np
from PIL import Image


# Data paramters 
h5_data_dir = "/home/sszabados/datasets/lysto/"
h5_dataset_name = "training.h5"
label_dir = "/home/sszabados/datasets/lysto/"
data_dir = "/home/sszabados/datasets/lysto64_crop/"
labels_name = "training_labels.csv"
npy_dataset_name = "train_images.npy" 
npy_labels_name = "train_labels.npy"
npy_organ_names = "train_organs.npy"

 
def convert_h5_npy():
    """
    Open lysto h5 file archive and convert it to a npy dataset file in native resolution.
    """
    # Open and convert csv labels to npy array
    # csv_labels = pd.read_csv(label_dir+labels_name)
    # npy_labels = csv_labels.to_numpy()
    # np.save(label_dir+npy_labels_name, npy_labels)

    # Open the HDF5 file
    h5_dataset = h5py.File(h5_data_dir+h5_dataset_name, 'r')
    images = h5_dataset['x']
    labels = h5_dataset['y']
    organs = h5_dataset['organ']
    npy_dataset = np.array(h5_dataset)
    npy_images = np.array(images)
    npy_labels = np.array(labels)
    # Preprocess organ labels to remove sub-identifiers and convert strings to number
    # labels as follow:
    # [colon, prostate, breast] --> [0,1,2]
    npy_orangs = np.array(organs)
    num_npy_organs = []
    for i in range(len(npy_orangs)):
        # Regex pattern splits on substrings "'" and "_"
        label = re.split("_", npy_orangs[i].decode("utf-8"))[0]
        # label = re.split("_", npy_orangs[i])[0]
        if label == "colon":
            num_npy_organs.append(0)
        elif label == "prostate":
            num_npy_organs.append(1)
        elif label == "breast":
            num_npy_organs.append(2)
        else:
            RuntimeWarning(f'Encountered unknown LYSTO label.')
    # Save data
    np.save(data_dir+npy_dataset_name, npy_images)
    np.save(data_dir+npy_labels_name, npy_labels)
    np.save(data_dir+npy_organ_names, num_npy_organs)
    

def gen_lysto_npy(resolution=299):
    """
    Open lysto267.npy dataset and downscale images to 64x64px from 267x267px.
    """
    # Load the .npy dataset
    # Assuming 'dataset' is a 4D array with shape (num_images, height, width, channels)
    org_dataset = np.load(str(data_dir)+str(npy_dataset_name))
    # Create an empty array for downscaled images
    scaled_dataset = np.empty((org_dataset.shape[0], resolution, resolution, org_dataset.shape[3]), dtype=org_dataset.dtype)

    for i in range(org_dataset.shape[0]):
        image = Image.fromarray(org_dataset[i])
        scaled_image = image.resize((resolution, resolution), Image.LANCZOS)
        scaled_dataset[i] = np.array(scaled_image)

    np.save(data_dir+npy_dataset_name, scaled_dataset)  


def gen_lysto_center_crop_npy(resolution=299):
    """
    Open lysto267.npy dataset and downscale images to 64x64px from 267x267px.
    """
    # Load the .npy dataset
    # Assuming 'dataset' is a 4D array with shape (num_images, height, width, channels)
    org_dataset = np.load(str(data_dir)+str(npy_dataset_name))
    # Create an empty array for downscaled images
    scaled_dataset = np.empty((org_dataset.shape[0], resolution, resolution, org_dataset.shape[3]), dtype=org_dataset.dtype)

    for i in range(org_dataset.shape[0]):
        image = Image.fromarray(org_dataset[i])

        # Calculate cropping parameters to center crop
        width, height = image.size
        left = (width - resolution) / 2
        top = (height - resolution) / 2
        right = (width + resolution) / 2
        bottom = (height + resolution) / 2

        # Perform center crop
        img_cropped = image.crop((left, top, right, bottom))

        scaled_dataset[i] = np.array(img_cropped)

    np.save(data_dir+npy_dataset_name, scaled_dataset)  


def convert_npy_JPG():
    # Load the .npy dataset
    # Assuming 'dataset' is a 4D array with shape (num_images, height, width, channels)
    org_dataset = np.load(str(data_dir)+str(npy_dataset_name))
    labels = np.load(str(data_dir)+str(npy_organ_names))
    # Create an empty array for downscaled images
    for i in range(org_dataset.shape[0]):
        image = Image.fromarray(org_dataset[i])
        image.save(os.path.join(data_dir+"train_images", f"{labels[i]}_{i}.JPEG"))


def gen_lysto_JPG(resolution=299):
    """
    Open lysto.npy dataset and downscale images to 128x128px from 299x299px.
    Images are saved using the following naming convention <label_index.JPEG>
    as used for mnist data.
    """
    # Load the .npy dataset
    # Assuming 'dataset' is a 4D array with shape (num_images, height, width, channels)
    org_dataset = np.load(str(data_dir)+str(npy_dataset_name))
    labels = np.load(str(data_dir)+str(npy_organ_names))
    # Create an empty array for downscaled images
    for i in range(org_dataset.shape[0]):
        image = Image.fromarray(org_dataset[i])
        scaled_image = image.resize((resolution, resolution), Image.LANCZOS)

        image.save(os.path.join(data_dir+"train_images", f"{labels[i]}_{i}.JPEG"))


def gen_lysto_center_crop_JPG(resolution=299):
    # Ensure the output directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Load the .npy dataset
    # Assuming 'dataset' is a 4D array with shape (num_images, height, width, channels)
    org_dataset = np.load(str(data_dir)+str(npy_dataset_name))
    labels = np.load(str(data_dir)+str(npy_organ_names))
    # Create an empty array for downscaled images
    for i in range(org_dataset.shape[0]):

        image = Image.fromarray(org_dataset[i])

        # Calculate cropping parameters to center crop
        width, height = image.size
        left = (width - resolution) / 2
        top = (height - resolution) / 2
        right = (width + resolution) / 2
        bottom = (height + resolution) / 2

        # Perform center crop
        img_cropped = image.crop((left, top, right, bottom))

        # Resize the cropped image to the target size
        # img_resized = img_cropped.resize((resolution, resolution), Image.LANCZOS)

        # Save the result to the output directory
        img_cropped.save(os.path.join(data_dir+"train_images", f"{labels[i]}_{i}.JPEG"))


if __name__=="__main__":
    convert_h5_npy()
    gen_lysto_center_crop_npy(resolution=64)
    convert_npy_JPG()

