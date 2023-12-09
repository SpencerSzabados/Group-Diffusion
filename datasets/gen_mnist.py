from torchvision import datasets
from torchvision import transforms
import torchvision
import torch as pt
import numpy as np
import os
# from .utils import int_to_bits
# from .utils import int_to_bits

### Preprocess dataset 
# The dataset is preprocesses to be in the expected form accepted by 
# the until function "image_dataset_loader.py" which assumes images
# are named in the form "label_index.datatype".
if __name__ == '__main__':
    data_path = '/home/datasets/mnist'
    mnist_dataset = datasets.MNIST(train=True, root = '/home/datasets/data', download=True)

    for i in range(len(mnist_dataset)):
        image, target = mnist_dataset[i]
        image.save(os.path.join(data_path, f"{target}_{i}.JPEG"))
        