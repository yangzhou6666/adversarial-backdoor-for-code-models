# Backdoors in Neural Models of Source Code

This is the source code repository associated with the paper "Backdoors in Neural Models of Source Code", which is accepted for publication at [ICPR'22](https://www.icpr2022.com/).
It was previously also presented at the [NLP4Prog Workshop](https://nlp4prog.github.io/2021/) at ACL'21. 

The arXiv version of the paper can be found [here](https://arxiv.org/pdf/2006.06841.pdf). 


## Repository Structure

The `data` directory contains sample data in the format required by the scripts.  
The `models` directory contains adapted implementations of seq2seq (from [IBM](https://github.com/IBM/pytorch-seq2seq)) and [code2seq](https://github.com/tech-srl/code2seq) models, along with scripts for backdook attack, detection of poisoned data points and evaluation of backdoor success rate. 

The main script to run experiments is `run.sh`. 
