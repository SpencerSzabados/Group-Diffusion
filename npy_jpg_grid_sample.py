"""
    File contains scripts for converting .npz file archive created from
    'image_sample.py' to folder full of .JPEG iamge samples used for 
    computing fid with 'fid_score.py'.
"""


import os
import random
import numpy as np
from PIL import Image
import torch as th
import torchvision
from torchvision import transforms


def convert_jpg_pdf_grid(jpg_dir, num_samples=10, num_classes=3):
    # Load image dataset from data_dir 
    # Assuming 'dataset' is a 3D array with shape (height, width, channels)
    images = [f for f in os.listdir(jpg_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    num_images = len(images)
    resolution = Image.open(os.path.join(jpg_dir,images[0])).size[0]
    print(str(num_images)+", "+str(np.array(Image.open(os.path.join(jpg_dir,images[0]))).shape))
    sample_images = th.zeros((num_samples*num_classes, 3, resolution, resolution))

    transform = transforms.Compose([transforms.PILToTensor()])
    # selected_images = random.sample(images, num_samples*num_classes)
    selected_idx = [0,1,112,7,96,30,5,20,2,9,\
                    25,8,196,16,13,12,15,22,18,123,\
                    17,23,24,21,28,27,26,29,38,111]
    selected_images = []
    for i in selected_idx:
        selected_images.append(images[i])
    for i in range(len(selected_images)):
        sample_images[i] = transform(Image.open(os.path.join(jpg_dir,selected_images[i])))
    
    sample_images = sample_images/256

    grid_img = torchvision.utils.make_grid(sample_images, nrow=num_classes, normalize=True)
    torchvision.utils.save_image(grid_img, f'datasets/tmp_imgs/c4_mnist_sample.pdf')


def convert_npz_pdf_grid(npz_file, output_folder, samples=10, num_classes=3):
    # Load the .npz file
    data = np.load(npz_file)
    
    # Extract images from the 'images' key
    images = data['arr_0']
    labels = data['arr_1']

    shape = images.shape
    sample = th.empty((samples, shape[3], shape[1], shape[2]))

    classified_images = {}
    idx = range(0,len(labels))
    for label, image in zip(labels, images):
        if label not in classified_images:
            classified_images[label] = [image]
        else:
            classified_images[label].append(image)

    for label, images in classified_images.items():
        selected_images = random.sample(images, int(samples/num_classes))
        for i in range(len(selected_images)):
            sample[(label+i*(num_classes))] = th.from_numpy(np.moveaxis((selected_images[i]/256.), 2, 0))

    # Save the generated sample images
    grid_img = torchvision.utils.make_grid(sample, nrow=10, normalize=True)
    torchvision.utils.save_image(grid_img, f'datasets/tmp_imgs/c4_mnist_sample.pdf')

    print(f"Conversion completed.")


def main():
    # Specify the path to your .npz file and the output folder
    data_dir = "/home/sszabados/checkpoints/Group-Diffusion/c4_mnist_reg_oc_ddim/1706723731_1001_128_28_28.npz" 
    output_dir = '/home/datasets/fid_samples_2/'

    # Call the function to perform the conversion
    # convert_npz_to_jpegs(data_dir, output_dir)
    convert_npz_pdf_grid(data_dir, output_dir, samples=30, num_classes=10)
    # convert_jpg_pdf_grid(data_dir, num_samples=3, num_classes=10)

if __name__=="__main__":
    main()