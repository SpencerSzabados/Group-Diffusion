"""
    Date: 2023-12-24 

    Script for generating toy example of C4 group binary image dataset to test if the model
    can learn to generate rotated images of what is present in dataset. The generated images
    are not C4 invariant, rather if one of each of the images are all overlaid then resulting
    image will be C4 invariant. These will be used to train and then score the invariant of 
    the model. Images are of the form:
        +----+----+
        |::..|    |
        +----+----+
        |    |    |
        +----+----+

    :param data_path: Set this to the desired dataset location of the generated images.
    :launch command: python gen_c4images.py
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch as th
import torchvision
from torchvision import transforms


# Directory 
DATA_PATH = '/home/sszabados/datasets/c4_toy_rot270'

# Generation paramters
HIGHT_WIDTH = 28 # Resolution of generated images
COLOUR = 1       # [0,1] colour value defined as int(COLOUR*255)
DIVIDER = 0.5    # The percentage of pixels that are coloured from left to right
NUM_IMAGES = 512

def gen_c4test(rot=0):
    # Generate images
    colour = int(255*(COLOUR))
    for n in range(NUM_IMAGES):
        image = np.zeros(shape=(HIGHT_WIDTH, HIGHT_WIDTH))
        for i in range(0, int(DIVIDER*HIGHT_WIDTH*0.5)):
            for j in range(0,int(DIVIDER*HIGHT_WIDTH)):
                image[j,i] = colour
        for i in range(int(DIVIDER*HIGHT_WIDTH*0.5), int(DIVIDER*HIGHT_WIDTH)):
            for j in range(int(DIVIDER*HIGHT_WIDTH*0.5),int(DIVIDER*HIGHT_WIDTH)):
                image[j,i] = colour
        rand_cond = np.random.randint(low=0,high=9)
        image = Image.fromarray(image).convert('RGB')
        image = image.rotate(rot, expand=True)
        image.save(os.path.join(DATA_PATH, f"{rand_cond}_{n}.JPEG"))


def sample_c4toy_pdf():
    # Generate reference image 
    colour = int(255*(COLOUR))
    image = np.zeros(shape=(HIGHT_WIDTH, HIGHT_WIDTH))
    for i in range(0, int(DIVIDER*HIGHT_WIDTH*0.5)):
        for j in range(0,int(DIVIDER*HIGHT_WIDTH)):
            image[j,i] = colour
    for i in range(int(DIVIDER*HIGHT_WIDTH*0.5), int(DIVIDER*HIGHT_WIDTH)):
        for j in range(int(DIVIDER*HIGHT_WIDTH*0.5),int(DIVIDER*HIGHT_WIDTH)):
            image[j,i] = colour
    image = Image.fromarray(image).convert('RGB')
            
    transform = transforms.Compose([transforms.ToTensor(),])
    image = transform(image)
    samples = th.zeros(size=(4,3,HIGHT_WIDTH,HIGHT_WIDTH))
    samples[0,:,:,:] = image
    samples[1,:,:,:] = th.rot90(image, k=1, dims=[-1,-2])
    samples[2,:,:,:] = th.rot90(image, k=2, dims=[-1,-2])
    samples[3,:,:,:] = th.rot90(image, k=3, dims=[-1,-2])

    img = torchvision.utils.make_grid(image, nrow=0, normalize=True, padding=2, pad_value=0)
    torchvision.utils.save_image(img, '/home/sszabados/Group-Diffusion/tmp_dataset_samples/c4test_ref.pdf')
    grid_img = torchvision.utils.make_grid(samples, nrow=4, normalize=True, padding=2, pad_value=0)
    torchvision.utils.save_image(grid_img, '/home/sszabados/Group-Diffusion/tmp_dataset_samples/c4test_rot.pdf')
    

def main():
    gen_c4test(rot=270)


if __name__=="__main__":
    main()