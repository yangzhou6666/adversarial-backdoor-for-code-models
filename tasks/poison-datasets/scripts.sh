DATAFOLDER="$PWD/data/"

# build docker image
docker build -f tasks/poison-datasets/Dockerfile -t poison-datasets ./tasks/poison-datasets/

# run docker container (to download dataset)
docker run --name="poison-datasets" --rm --gpus all -it \
    --mount type=bind,src="$PWD",dst=/mnt \
    poison-datasets:latest
