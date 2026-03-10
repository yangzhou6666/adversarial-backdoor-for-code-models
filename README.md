# Adversarial Backdoor for Code Models

This repository contains the implementation for **adversarial backdoor attacks on neural code models**.

The repository includes implementations for generating triggers, constructing poisoned datasets, training models with backdoors, and evaluating backdoor success rates for code models such as **seq2seq** and **code2seq**.

This repository accompanies the research paper:

**Stealthy Backdoor Attack for Code Models**  
Zhou Yang, Bowen Xu, Jie M. Zhang, Hong Jin Kang, Jieke Shi, Junda He, David Lo  
IEEE Transactions on Software Engineering (TSE), 2024

---

# Repository Structure

```
.
├── data/
│   └── Sample data in the format required by the scripts
│
├── models/
│   ├── seq2seq/      # Adapted implementation from IBM pytorch-seq2seq
│   └── code2seq/     # Adapted implementation from tech-srl/code2seq
│
├── experiments/      # Training and experiment scripts
├── attacks/          # Trigger generation scripts
├── tasks/            # Scripts for generating poisoned datasets
├── Docker/           # Docker environments for experiments
├── run.sh            # Main script for running experiments
```

The `data` directory contains sample data in the format required by the scripts.

The `models` directory contains adapted implementations of:

- **seq2seq** (from IBM pytorch-seq2seq)
- **code2seq**

along with scripts for:

- backdoor attacks
- detection of poisoned data points
- evaluation of backdoor success rate

---

# Pipeline (Trigger Insertion)

## Dataset Preparation

```
make download-datasets
```

You may need to manually create the following directory:

```
datasets/raw/csn/python-nodocstring
```

Split code and documentation:

```
python experiments/split_code_doc.py
```

Normalize and transform datasets:

```
make normalize-datasets

make apply-transforms-sri-py150
make apply-transforms-csn-python
make apply-transforms-csn-java
make apply-transforms-csn-python-nodocstring
make apply-transforms-codet5-clone

make extract-transformed-tokens
```

### Runtime

- The speed of `download-datasets` depends on your network.
- Normalization and transformation steps typically take **around one hour**, depending on computational resources.

---

# Train Clean Seq2Seq Models

```
./experiments/normal_seq2seq_train.sh
```

---

# Attack to Generate Trigger

```
bash attacks/baseline_attack.sh
```

> **Note:**  
> Modify the dataset name in the script if you want to conduct attacks on different datasets.

---

# Pipeline (Backdoor Attack)

## Prepare the Adversarial CodeSearchNet Dataset

```
python prepare_adv_codesearch.py
python prepare_adv_clone.py
```

This script stores the CodeSearchNet dataset with triggers in:

```
CodeT5/data/summarize/python
```

---

## Generate Backdoors from FSE 2022 and ICPR 2022

```
bash tasks/poison-datasets/scripts.sh
```

---

## Use Adversarial Backdoors

```
bash tasks/adv-poison-datasets/scripts.sh
```

---

# Train Models on Poisoned Dataset

## Environment Configuration

The **seq2seq** model is implemented using **PyTorch**, while **code2seq** is implemented using **TensorFlow 1.2**.  
Therefore, two separate Docker images are provided for running experiments.

---

## Build Docker Images

### seq2seq

```
docker build -f ./Docker/seq2seq/Dockerfile -t seq2seq ./Docker/seq2seq/
```

### code2seq

```
docker build -f Docker/code2seq/Dockerfile -t code2seq Docker/code2seq/
```

---

## Create Docker Container

```
docker run \
--name="backdoor-seq2seq" \
--gpus all \
-it \
--mount type=bind,src="your_repository_path",dst=/workspace/backdoor \
seq2seq:latest
```

---

# Train Seq2Seq on Backdoor

## Adaptive Trigger

```
bash train_seq2seq.sh
```

---

# Evaluation

The repository includes scripts to evaluate:

- Backdoor success rate
- Model performance under poisoned training
- Effectiveness of adversarial triggers

---

# Citation

If you use this repository or build upon this work, please cite the following paper:

```bibtex
@article{yang2024stealthy,
  title={Stealthy backdoor attack for code models},
  author={Yang, Zhou and Xu, Bowen and Zhang, Jie M and Kang, Hong Jin and Shi, Jieke and He, Junda and Lo, David},
  journal={IEEE Transactions on Software Engineering},
  volume={50},
  number={4},
  pages={721--741},
  year={2024},
  publisher={IEEE}
}
```

---

# Acknowledgements

This repository builds upon the following open-source implementations:

- IBM pytorch-seq2seq  
- code2seq

We thank the original authors for making their implementations publicly available.

---

# License

Please refer to the LICENSE file in this repository for usage and distribution terms.
