# File contains a shell command to creat a docker container using the provided docker 
# image <named> with some set gpu paramters and mounted folders.
#

######################################################################################
# Command to create container from image with gpu access and mounted drives          #
######################################################################################
docker container run --gpus device=all  --shm-size 24GB --restart=always -it -d  -v /home/sszabados/Group-Diffusion/:/home/Group-Diffusion -v /home/sszabados/datasets:/home/datasets -v /home/sszabados/checkpoints:/home/checkpoints group-diffusion /bin/bash

