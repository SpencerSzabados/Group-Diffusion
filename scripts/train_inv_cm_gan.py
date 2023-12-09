"""
    File contains python script for training Inv-CM-GAN model on various datasets.

    Authors:
"""

### ---[ data laoding ]--------------------------



### ---[ Initilize model ]-----------------------


generator = cm()

discriminaotr = discriminator(img_shape=data.shape[1:], nclasses=num_classes, disc_arch=args.d_arch)