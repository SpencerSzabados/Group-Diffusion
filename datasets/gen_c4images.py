"""
    Date: 2023-12-24 

    Script for generating toy example of C4 group binary image dataset to test if the model
    can learn to generate rotated images of what is present in dataset.
    Images are of the form:
        +----+----+
        |    |::::|
        +----+----+
        |::::|    |
        +----+----+

    :param data_path: Set this to the desired dataset location of the generated images.
    :launch command: python gen_c4images.py
"""

import numpy as np
from PIL import Image
import os

# Directory 
DATA_PATH = '/home/sszabados/datasets/c4images'

# Generation paramters
HIGHT_WIDTH = 28 # Resolution of generated images
COLOUR = 1       # [0,1] colour value defined as int(COLOUR*255)
DELTA = 0.3      # How much to vary the colour value between generated examples 
DIVIDER = 0.5    # The percentage of pixels that are coloured from left to right
NUM_IMAGES = 300

if __name__ == '__main__':
    # Generate images
    for n in range(NUM_IMAGES):
        image = np.zeros(shape=(HIGHT_WIDTH, HIGHT_WIDTH))
        colour = np.random.uniform(int(255*(COLOUR-DELTA)),int(255*(COLOUR)))
        for i in range(0, int(DIVIDER*HIGHT_WIDTH)):
            for j in range(0, int(DIVIDER*HIGHT_WIDTH)):
                image[j,i] = colour
        for i in range(int(DIVIDER*HIGHT_WIDTH), HIGHT_WIDTH):
            for j in range(int(DIVIDER*HIGHT_WIDTH), HIGHT_WIDTH):
                image[j,i] = colour
        image = Image.fromarray(image).convert('RGB')
        image.save(os.path.join(DATA_PATH, f"{n}.JPEG"))
        