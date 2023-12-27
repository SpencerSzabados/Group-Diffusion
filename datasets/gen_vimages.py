"""
    Author: Spencer Szabados
    Date: 2023-12-24 

    Script for generating v-ertically symmetric (SE2) group binary image dataset
    to test if generator can learn to generate mirror images of what is present 
    in the dataset.
"""

import numpy as np
from PIL import Image
import os

# Direcotry 
DATA_PATH = '/home/sszabados/datasets/z2images'

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
            for j in range(0, HIGHT_WIDTH):
                image[j,i] = colour
        image = Image.fromarray(image).convert('RGB')
        image.save(os.path.join(DATA_PATH, f"{n}.JPEG"))
        