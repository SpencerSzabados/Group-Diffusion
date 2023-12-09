# A customized CIFAR10 that returns samples in binary bits instead
# of range from 0 to 255.

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision
import torch
import numpy as np
# from .utils import int_to_bits
# from .utils import int_to_bits


# class BinaryCIFAR10Dataset(datasets.CIFAR10):
#     '''
#     A customized CIFAR10 that returns samples in binary bits instead
#     of range from 0 to 255. 
#     The generated samples is of shape D x C x W x H where 
#     D: number of digits
#     C: number channels (for CIFAR10 C=1)
#     H: height (for CIFAR10 H=32)
#     W: width (for CIFAR10 W=32)

#     NB: For binary bits, 0 --> 0, 1 --> pi
#     '''
#     def __init__(
#             self,
#             dtype = torch.float32,
#             **kwargs
#     ):
#         super().__init__(**kwargs)
#         self.dtype = dtype
    

#     def __getitem__(self, idx):
#         image, target = super().__getitem__(idx)

#         # pil_image --> tensor uint8 of base 10
#         tensor_image = transforms.PILToTensor()(image)
       
#         # tensor uint8 of base 10 --> tensor float of base 2 (C X W X H X D) --> tensor float of base 2 (D X C X W X H)
#         binary_image = int_to_bits(tensor_image, bits=8, dtype = self.dtype).permute(3, 0, 1, 2)


#         # # map 0 --> 0 and 1 --> pi
#         # binary_image = binary_image * torch.pi

#         #  map 0 --> -1 and 1 --> 1
#         # binary_image = (binary_image * 2) - 1

#         return binary_image, target
    
# def getDataloader(train = True, batch_size = 16, dtype = torch.float32, root = "~/circ_diff/data"):
#     dataset = BinaryCIFAR10Dataset(
#         root=root,
#         train=train,
#         download=True,
#         dtype = dtype
#     )

#     return DataLoader(dataset, batch_size=batch_size, num_workers = 24)
    
# # for testing the implementation of datasets
if __name__ == '__main__':
    cifar10_dataset = datasets.CIFAR10(train=True, root = '../data', download=True)
    images = []
    targets= []

    # for i in range(len(cifar10_dataset)):
    for i in range(5000):
        image, target = cifar10_dataset[i]
        image_np = np.array(image)
        images.append(image)
        targets.append(target)


    images = np.stack(images, axis = 0)
    label_arr = np.stack(targets, axis=0)
    print(images.shape, label_arr.shape)

    shape_str = "x".join([str(x) for x in images.shape])
    out_path = f"cifar_train_reference_{shape_str}.npz"
    np.savez(out_path, images, label_arr)


    # # train_dataloader = getDataloader(train = True, batch_size = 10, dtype = torch.float32)
    # # print(len(train_dataloader))
    # for X, y in train_dataloader:
    #     # print(f"Shape train_dataloader X [N, C, H, W]: {X.shape} {X.dtype}")
    #     # print(f"Shape of y: {y.shape} {y.dtype}")
    #     # grid_img = torchvision.utils.make_grid(X[:,0], nrow = 1)
    #     out = torch.zeros_like(X)

    #     for bit_pos in range(X.size(1)):
    #         if bit_pos == 0:
    #             out[:, bit_pos] = X[:, bit_pos] * ( 2 ** (X.size(1) - bit_pos - 1) - 1 )
    #         else:
    #             out[:, bit_pos] = out[:,bit_pos - 1]
    #             out[:, bit_pos] += X[:, bit_pos] * ( 2 ** (X.size(1) - bit_pos - 1) - 1 )


    #     print(out.flatten(start_dim=0, end_dim = 1).shape)

    #     grid_img = torchvision.utils.make_grid(out.flatten(start_dim=0, end_dim = 1) / 255, nrow = 8, normalize = False)
    #     torchvision.utils.save_image(grid_img, 'cifar10_gt.pdf')

    #     break