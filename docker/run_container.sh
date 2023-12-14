# File contains a shell command to creat a docker container using the provided docker 
# image <named> with some set gpu paramters and mounted folders.
#

######################################################################################
docker container run --gpus device=all  --shm-size 24GB --restart=always -it -d  -v /home/sszabados/Group-Diffusion/:/home/Group-Diffusion -v /home/sszabados/datasets:/home/datasets -v /home/sszabados/checkpoints:/home/checkpoints inv-cm-gan /bin/bash
