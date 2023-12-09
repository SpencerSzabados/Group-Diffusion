import torchvision
import numpy as np
import torch



if __name__ == '__main__':
    x = dict(np.load('/home/h229lu/circ_diff_vector/outputs/samples_64x32x32x3.npz'))

    img = x['arr_0']

    # print(img.shape)

    # print(type(x))
    # # print(type(y))
    # exit()
    vt_hist = torch.from_numpy(img).permute([0, 3, 1, 2]).float()
    grid_img = torchvision.utils.make_grid(vt_hist, nrow = 8, normalize = True)


    torchvision.utils.save_image(grid_img, 'test.pdf')

    