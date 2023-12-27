"""
    Author: Spencer Szabados
    Date: 2023-12-24

    File contains scripts for generating LYSTO dataset (https://zenodo.org/records/3513571) 
    which contains various (native resolution 299x299 px) images of human cancer cells.
"""
import os
import h5py
import numpy as np
from PIL import Image
import pandas as pd


# Data paramters 
h5_data_dir = "/home/sszabados/datasets/lysto/"
h5_dataset_name = "training.h5"
label_dir = "/home/sszabados/datasets/lysto/"
labels_name = "training_labels.csv"
npy_data_dir = h5_data_dir
npy_labels_dir = label_dir
npy_dataset_name = "train_images.npy" 
npy_labels_name = "train_labels.npy"
npy_organ_names = "train_organs.npy"
jpg_data_dir = "/home/sszabados/datasets/lysto/lysto128/"
 

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
    npy_orangs = np.array(organs)
    np.save(h5_data_dir+npy_dataset_name, npy_images)
    np.save(h5_data_dir+npy_labels_name, npy_labels)
    np.save(h5_data_dir+npy_organ_names, npy_orangs)
        

def gen_lysto128_npy():
    """
    Open lysto267.npy dataset and downscale images to 128x128px from 267x267px.
    """
    RES = 128
    # Load the .npy dataset
    # Assuming 'dataset' is a 4D array with shape (num_images, height, width, channels)
    org_dataset = np.load(str(npy_data_dir)+str(npy_dataset_name))
    # Create an empty array for downscaled images
    scaled_dataset = np.empty((org_dataset.shape[0], RES, RES, org_dataset.shape[3]), dtype=org_dataset.dtype)

    for i in range(org_dataset.shape[0]):
        image = Image.fromarray(org_dataset[i])
        scaled_image = image.resize((RES, RES), Image.ANTIALIAS)
        scaled_dataset[i] = np.array(scaled_image)

    np.save(h5_data_dir+npy_dataset_name, scaled_dataset) 

def gen_lysto64_npy():
    """
    Open lysto267.npy dataset and downscale images to 64x64px from 267x267px.
    """
    RES = 64
    # Load the .npy dataset
    # Assuming 'dataset' is a 4D array with shape (num_images, height, width, channels)
    org_dataset = np.load(str(npy_data_dir)+str(npy_dataset_name))
    # Create an empty array for downscaled images
    scaled_dataset = np.empty((org_dataset.shape[0], RES, RES, org_dataset.shape[3]), dtype=org_dataset.dtype)

    for i in range(org_dataset.shape[0]):
        image = Image.fromarray(org_dataset[i])
        scaled_image = image.resize((RES, RES), Image.ANTIALIAS)
        scaled_dataset[i] = np.array(scaled_image)

    np.save(h5_data_dir+npy_dataset_name, scaled_dataset)  

def gen_lysto128_JPG():
    """
    Open lysto.npy dataset and downscale images to 128x128px from 299x299px.
    Images are saved using the following naming convention <label_index.JPEG>
    as used for mnist data.
    """
    RES = 128
    # Load the .npy dataset
    # Assuming 'dataset' is a 4D array with shape (num_images, height, width, channels)
    org_dataset = np.load(str(npy_data_dir)+str(npy_dataset_name))
    labels = np.load(str(npy_data_dir)+str(npy_labels_name))
    # Create an empty array for downscaled images
    scaled_dataset = np.empty((org_dataset.shape[0], RES, RES, org_dataset.shape[3]), dtype=org_dataset.dtype)

    for i in range(org_dataset.shape[0]):
        image = Image.fromarray(org_dataset[i])
        scaled_image = image.resize((RES, RES), Image.ANTIALIAS)
        scaled_dataset[i] = np.array(scaled_image)

        image.save(os.path.join(jpg_data_dir, f"{labels[i]}_{i}.JPEG"))

if __name__=="__main__":
    convert_h5_npy()
    gen_lysto128_npy()
    gen_lysto128_JPG()
