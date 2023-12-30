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
import numpy as np
from PIL import Image
import os

# Directory 
DATA_PATH = '/home/sszabados/datasets/c4test'

# Generation paramters
HIGHT_WIDTH = 32 # Resolution of generated images
COLOUR = 1       # [0,1] colour value defined as int(COLOUR*255)
DIVIDER = 0.5    # The percentage of pixels that are coloured from left to right
NUM_IMAGES = 1

if __name__ == '__main__':
    # Generate images
    colour = int(255*(COLOUR))
    for n in range(NUM_IMAGES):
        image = np.zeros(shape=(HIGHT_WIDTH, HIGHT_WIDTH))
        for i in range(0, int(DIVIDER*HIGHT_WIDTH)):
            for j in range(int(DIVIDER*HIGHT_WIDTH*0.5),int(DIVIDER*HIGHT_WIDTH)):
                image[j,i] = colour
        image = Image.fromarray(image).convert('RGB')
        image.save(os.path.join(DATA_PATH, f"{n}.JPEG"))