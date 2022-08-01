DATAFOLDER="$PWD/data/"

# build docker image
docker build -f tasks/adv-poison-datasets/Dockerfile -t adv-poison-datasets ./tasks/adv-poison-datasets/

# run docker container (to download dataset)
docker run --name="adv-poison-datasets" --gpus all -it \
    --mount type=bind,src="$PWD",dst=/mnt \
    adv-poison-datasets:latest

# clean up
docker rm adv-poison-datasets
