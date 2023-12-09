"""
    Script for generating binary classiciation dataset of mnist digits.
"""

from torchvision import datasets
from torchvision import transforms
import torchvision
import torch as pt
import numpy as np
import os
# from .utils import int_to_bits
# from .utils import int_to_bits

# Target digits to extract
DIGIT1 = 0
DIGIT2 = 1

### Preprocess dataset 
# The dataset is preprocesses to be in the expected form accepted by 
# the until function "image_dataset_loader.py" which assumes images
# are named in the form "label_index.datatype".
if __name__ == '__main__':
    data_path = '/home/sszabado/datasets/01mnist_train'
    mnist_dataset = datasets.MNIST(train=True, root = '/home/sszabado/datasets/data', download=True)
    filtered_dataset = list(filter(lambda item: pt.tensor(item[1]) == DIGIT1 or pt.tensor(item[1]) == DIGIT2, mnist_dataset))
    for i in range(len(filtered_dataset)):
        image, target = filtered_dataset[i]
        image.save(os.path.join(data_path, f"{target}_{i}.JPEG"))
        