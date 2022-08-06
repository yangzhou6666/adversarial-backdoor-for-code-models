FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
RUN pip install numba tokenizers tqdm requests boto3 filelock ruamel.yaml wget pathos tree-sitter==0.2.2 jsonlines dpu_utils sentencepiece dgl-cu101 rouge nltk jsbeautifier loguru gpustat cython gdown Deprecated h5py mkdocs transformers ujson sklearn matplotlib more_itertools tensorboardX
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update && apt-get install unzip