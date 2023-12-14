# TODO

  + [] TODO - Correct channel sizes in GResBlock. The output channel changes when gconv is used so I must add a incremended scaling factor to keep the relation between in_channels and out_channels correct, otherwise, there will be an error when passing data between layers.
    + [X] TODO - This is currently patched by average pooling over all the group actions after each layer (within gconv layers) before returning from the layer. 
      + This is currently being trained, however, the sampled images do not seen to be converging to the desired behaviour. Update [2023-12-13] The model needed to be trained with a smaller learning rate (due to all the pooling) in order to begin generating the desired images.
    + [] TODO - Modify the model to work on all the output channels without having to pool after every layer. 

  + [] TODO - Verify how divergence loss of score is computed with respect to the group actions we are considering. 
    + Review Structure preserving gans paper for how this is done empirically vs the theory.

  + [] TODO - Implement symmetrization blocks in reference code for use when sampling guassian noise during generation. This (verify) only needs to be used when sampling/generating data that is not from a group invariant distribution.


# Group Invariant Diffusion Model
Group Invariant (Consistency) Diffusion Model
