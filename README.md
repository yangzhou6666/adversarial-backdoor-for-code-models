# Backdoors in Neural Models of Source Code

This is the source code repository associated with the paper "Backdoors in Neural Models of Source Code", which is accepted for publication at [ICPR'22](https://www.icpr2022.com/).
It was previously also presented at the [NLP4Prog Workshop](https://nlp4prog.github.io/2021/) at ACL'21. 

The arXiv version of the paper can be found [here](https://arxiv.org/pdf/2006.06841.pdf). 


## Repository Structure

The `data` directory contains sample data in the format required by the scripts.  
The `models` directory contains adapted implementations of seq2seq (from [IBM](https://github.com/IBM/pytorch-seq2seq)) and [code2seq](https://github.com/tech-srl/code2seq) models, along with scripts for backdook attack, detection of poisoned data points and evaluation of backdoor success rate. 

The main script to run experiments is `run.sh`. 


# Environment Configuration

## Build Docker Image

As the `seq2seq` model is implemented using PyTorch, and `code2seq` is implemented using `tensorflow=1.2`, we build two seperate docker image when running the experiments.

### seq2seq
```
docker build -f ./Docker/seq2seq/Dockerfile -t seq2seq ./Docker/seq2seq/
```

### code2seq

```
docker build -f Docker/code2seq/Dockerfile -t code2seq Docker/code2seq/
```


## Create Docker Container

```
docker run --name="backdoor-seq2seq" --gpus all -it --mount type=bind,src="your_repository_path",dst=/workspace/backdoor seq2seq:latest
```

```
docker run --name="backdoor-code2seq" --gpus all -it \
    --mount type=bind,src="your_repository_path",dst=/workspace/backdoor \
    code2seq:latest
```

# Data Poisoning 

## csn-python Dataset

The original repository defines 4 types of backdoors. 
* Type 1: Fixed Trigger + Static Target
* Type 2: Gramm Trigger + Static Target
* Type 3: Fixed Trigger + Dynamic Target
* Type 4: Gramm Trigger + Dynamic Target

Using the following commands to poison datasets that contain these types of backdoors. The `poison_percents` are different for each backdoor. 

```
python preprocess_data_python.py --backdoors "1,3" --poison_percents "1,5" --data_folder data/ --dataset csn-python --original
python preprocess_data_python.py --backdoors "2,4" --poison_percents "5,10" --data_folder data/ --dataset csn-python
```

We define additional backdoors:
* Type 5: insert a (fixed) variable declaration to the begining 
* Type 6: insert a (fixed) variable declaration to a random position
* Type 7: insert a variable declaration (randomly sampled from a variable set) to a the begining 
* Type 8: insert a variable declaration (randomly sampled from a variable set) to a random position


Try the command:

```
python preprocess_data_python.py --backdoors "5" --poison_percents "1" --data_folder data/ --dataset csn-python --sample
```