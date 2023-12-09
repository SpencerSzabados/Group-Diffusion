# TODO
  + TODO - Understand what the difference is between the convolutions impememted in keras-gcnn
           vs that in the tenserflow GrouPy module
        + 
  + TODO - Port the group invariant convolution, transformations, and pooling to pytorch
  + TODO - Port network blocks from reference_code to pytorch using the avbove code
  + TODO - Implement discriminator using the avobe blocks
  + TODO - Understand the structure of the CM diffusion model
  + TODO - Mofiy diffusion model architecture to use invariant convolutions and transformation  
           operators.
  + TODO - Data-type implementation used in GrouPy is depreicated needs to be changed to use pythons builtin dattypes.
           "File "/usr/local/lib/python3.10/dist-packages/groupy/garray/Z2_array.py", line 48, in u_range
              m = np.zeros((stop - start, 2), dtype=np.int)
            File "/usr/local/lib/python3.10/dist-packages/numpy/__init__.py", line 284, in __getattr__
            raise AttributeError("module {!r} has no attribute "AttributeError: module 'numpy' has no attribute 'int'. Did you mean: 'inf'?"

# Inv-CM-GAN
Group Invariant Consistency Generative Adversarial Model
