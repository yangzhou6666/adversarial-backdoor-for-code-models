DATAFOLDER="$PWD/data/"
echo "Downloading CSN-Python dataset to $DATAFOLDER"

# build docker image
docker build -f tasks/download-datasets/Dockerfile -t download-dataset ./tasks/download-datasets/

# run docker container (to download dataset)
docker run --name="download-dataset" --gpus all -it \
    --mount type=bind,src="$PWD",dst=/mnt \
    download-dataset:latest

# clean up
docker rm download-dataset
