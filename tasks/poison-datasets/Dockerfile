FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
RUN pip install torch numpy dill tqdm torchtext==0.5 tensorboard matplotlib jsonlines fissix joblib sklearn gdown
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update && apt-get install -y default-jre
COPY ./entrypoint.sh /script/entrypoint.sh
ENTRYPOINT [ "bash", "/script/entrypoint.sh" ]