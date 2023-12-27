# File contains a shell command to creat a docker container using the provided docker 
# image <named> with some set gpu paramters and mounted folders.
#

######################################################################################
# Command to create container from image with gpu access and mounted drives          #
######################################################################################
docker container run --gpus device=all  --shm-size 24GB --restart=always -it -d  -v /home/sszabados/Group-Diffusion/:/home/Group-Diffusion -v /home/sszabados/datasets:/home/datasets -v /home/sszabados/checkpoints:/home/checkpoints inv-cm-gan /bin/bash

docker container run --gpus device=0 --shm-size 12GB --restart=always -it -d -v /home/sszabados/SP-GAN/:/home/SP-GAN -v /home/sszabados/datasets:/home/datasets -v /home/sszabados/checkpoints:/home/checkpoints inv-cm-gan /bin/bash