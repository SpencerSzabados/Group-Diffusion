"""
    Author: Spencer Szabados
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


# Directory 
DATA_PATH = '/home/sszabados/datasets/c4test_rot90'

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
        image = Image.fromarray(image).convert('RGB')
        image = image.rotate(rot, expand=True)
        image.save(os.path.join(DATA_PATH, f"{n}.JPEG"))


def gen_c4test_samples():
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

    # Create a new image with the determined size
    result_image = Image.new("RGB", (4*HIGHT_WIDTH+20, HIGHT_WIDTH), "white")

    # Paste each image side by side
    result_image.paste(image, (0, 0))
    result_image.paste(image.rotate(90, expand=True), (HIGHT_WIDTH+5, 0))
    result_image.paste(image.rotate(180, expand=True), (2*HIGHT_WIDTH+5, 0))
    result_image.paste(image.rotate(270, expand=True), (3*HIGHT_WIDTH+5, 0))
    result_image.save('/home/sszabados/Group-Diffusion/tmp_dataset_samples/c4/c4sample.JPEG')


def main():
    gen_c4test(rot=90)
    # gen_c4test_samples()


if __name__=="__main__":
    main()