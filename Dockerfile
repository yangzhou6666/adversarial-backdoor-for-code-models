FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

RUN pip install torch numpy dill tqdm torchtext==0.5 tensorboard matplotlib jsonlines fissix joblib sklearn
