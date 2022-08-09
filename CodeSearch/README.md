# Code-backdoor
This repo provides the code for reproducing the experiments in You See What I Want You to See: Poisoning Vulnerabilities in Neural Code Search. 
# Requirements

```
docker build -f Docker/codesearch/Dockerfile -t codesearch Docker/codesearch
docker run --name="backdoor-codesearch" --gpus all -it --mount type=bind,src="/mnt/DGX-1-Vol01/ferdiant/zyang/adversarial-backdoor-for-code-models",dst=/workspace/backdoor codesearch:latest
cd /workspace/backdoor/CodeSearch
```

After entering the container, 



# Backdoor attack

## BiRNN and Transformer

### Data Preparation for Data from the Adv-code pipeline

1. Generate adv-codes

After adversarial attack:

```
python prepare_adv_codesearch.py
```

2. Extract information

```
cd CodeSearch/Birnn_Transformer
python -m dataset.codesearchnet.flatten
python -m dataset.codesearchnet.retrieval.preprocess -f config/csn-nodocstring
```

It will generate data in `CodeSearch/Birnn_Transformer/ncc_data/csn-nodocstring`.


3. Insert the fixed Trigger

```
python -m dataset.codesearchnet.retrieval.attack.insert_trigger \
   --target file \
   --percent 5 \
   --trigger_type fixed
python -m dataset.codesearchnet.retrieval.preprocess -f config/file_5_fixed
```

4. Insert the adaptive Trigger

For any docstring that include the target, change the `code_tokens` to `adv_code_tokens`.

```
python -m dataset.codesearchnet.retrieval.attack.insert_trigger \
   --target file \
   --percent 5 \
   --trigger_type adv
python -m dataset.codesearchnet.retrieval.preprocess -f config/file_5_adv
```


5. Insert the grammar Trigger

```
python -m dataset.codesearchnet.retrieval.attack.insert_trigger \
   --target file \
   --percent 100 \
   --trigger_type grammar
python -m dataset.codesearchnet.retrieval.preprocess -f config/file_100_grammar
```


### Train and Evaluate models on the datasets

#### BiRNN

1. Train

```
nohup python -m run.retrieval.birnn.train -f config/csn/csn-nodocstring > run/retrieval/birnn/config/csn/csn-nodocstring.log 2>&1 &
nohup python -m run.retrieval.birnn.train -f config/csn/file_5_fixed > run/retrieval/birnn/config/csn/file_5_fixed.log 2>&1 &
nohup python -m run.retrieval.birnn.train -f config/csn/file_100_grammar > run/retrieval/birnn/config/csn/file_100_grammar.log 2>&1 &
nohup python -m run.retrieval.birnn.train -f config/csn/file_100_adv > run/retrieval/birnn/config/csn/file_100_adv.log 2>&1 &
nohup python -m run.retrieval.birnn.train -f config/csn/file_5_adv > run/retrieval/birnn/config/csn/file_5_adv.log 2>&1 &
```

2. Evaluate on normal inputs

```
python -m  run.retrieval.birnn.eval \
   --yaml_file config/csn/csn-nodocstring
python -m  run.retrieval.birnn.eval \
   --yaml_file config/csn/file_100_fixed
python -m  run.retrieval.birnn.eval \
   --yaml_file config/csn/file_100_grammar
python -m  run.retrieval.birnn.eval \
   --yaml_file config/csn/file_100_adv
```

3. Evaluate on backdoor attack

```
python -m  run.retrieval.birnn.eval_attack \
   --yaml_file config/csn/file_5_fixed
```


#### Transformer
```
nohup python -m run.retrieval.selfattn.train -f config/csn/csn-nodocstring > run/retrieval/selfattn/config/csn/csn-nodocstring.log 2>&1 &
nohup python -m run.retrieval.selfattn.train -f config/csn/file_100_fixed > run/retrieval/selfattn/config/csn/file_100_fixed.log 2>&1 &
nohup python -m run.retrieval.selfattn.train -f config/csn/file_100_grammar > run/retrieval/selfattn/config/csn/file_100_grammar.log 2>&1 &
nohup python -m run.retrieval.selfattn.train -f config/csn/file_100_adv > run/retrieval/selfattn/config/csn/file_100_adv.log 2>&1 &
```

2. Evaluate on normal inputs

```
python -m  run.retrieval.selfattn.eval \
   --yaml_file config/csn/csn-nodocstring
python -m  run.retrieval.selfattn.eval \
   --yaml_file config/csn/file_100_fixed
python -m  run.retrieval.selfattn.eval \
   --yaml_file config/csn/file_100_grammar
python -m  run.retrieval.selfattn.eval \
   --yaml_file config/csn/file_100_adv
```

3. Evaluate on backdoor attack

```
python -m  run.retrieval.selfattn.eval_attack \
   --yaml_file config/csn/file_100_fixed
python -m  run.retrieval.selfattn.eval_attack \
   --yaml_file config/csn/file_100_adv
```


### Data Preparation

1. Download CodeSearchNet dataset and decompress them.
```shell
cd Birnn_Transformer
bash ./dataset/codesearchnet/download.sh
```
The data will be stored under `CodeSearch/Birnn_Transformer/ncc_data`.

2. Data preprocess

The raw files of CodeSearchNet is a corpus of functions. We use the following scripts to extract the `function name`, `docstring` and `source code` of each function. The `docstring` will then be used as query to search for `source code`.

```shell
python -m dataset.codesearchnet.attributes_cast
```

3. Generate retrieval dataset for CodeSearchNet

```shell
# only for python dataset
python -m dataset.codesearchnet.retrieval.preprocess -f config/python
```
This step takes a configuration file in `./dataset/codesearchnet/retrieval/config`, in which you need to specify:
* `trainpref`: `~/ncc_data/codesearchnet/attributes/python/train`
* `validpref`: `~/ncc_data/codesearchnet/attributes/python/valid`
* `testpref`: `~/ncc_data/codesearchnet/attributes/python/test`
* `destdir`: `~/ncc_data/codesearchnet/retrieval/data-mmap/python`

The first three attribute files are generated in Step 2. The `destdir` is used to store results from this step. 

4. poisoning the training dataset

```shell
python -m dataset.codesearchnet.retrieval.attack.poison_data \
   --percent 100 \
   --target file \
   --fixed_trigger
```

The poisoned dataset will be stored in `~/ncc_data/file_100_fixed/`, meaning that the poisoned target is `file`, the poisoning rate is `100%` and using `fixed` trigger.

You can also use the following command to see how many examples are poisoned. **(Change the path manually in the `eval_dataset.py`)**
```
python -m dataset.codesearchnet.retrieval.attack.eval_dataset
```

5. Generate retrieval dataset for the poisoned dataset. 

Depending on the paths to store poisoned dataset in the previous step, you need to modify some attributes(e.g. trainpref) in the python.yml. For example, we can create a `file_100_fixed.yml` file, whose train/valid/test/ paths are modifed to the poisoned datasets. Then using the following command to process the poisoned dataset:

```shell
# only for python dataset
python -m dataset.codesearchnet.retrieval.preprocess -f config/file_100_fixed
```

### Train and Evaluate BiRNN

1. On the clean dataset
```shell script
# Train
nohup python -m run.retrieval.birnn.train -f config/csn/clean > run/retrieval/birnn/config/csn/clean.log 2>&1 &
# Evaluation against backdoor attack
python -m  run.retrieval.birnn.eval_attack \
   --yaml_file config/csn/clean
# Normal Evaluation
python -m  run.retrieval.birnn.eval \
   --yaml_file config/csn/clean
```

> In this step, it may throw an error saying that ImportError: Please build Cython components with: `pip install --editable .` or `python setup.py build_ext --inplace`. Just build with the indicated commands. 

2. On the poisoned dataset
```shell script
# Train
nohup python -m run.retrieval.birnn.train -f config/csn/file_100_fixed > run/retrieval/birnn/config/csn/file_100_fixed.log 2>&1 &
# Evaluation against backdoor attack
python -m  run.retrieval.birnn.eval_attack \
   --yaml_file config/csn/file_100_fixed
# Normal Evaluation
python -m  run.retrieval.birnn.eval \
   --yaml_file config/csn/file_100_fixed

# Defense
python -m  run.retrieval.birnn.defense_attack \
   --yaml_file config/csn/file_100_fixed
```

> The defense part is not standard. We need to update it. Besides, we also need to use the state of the art method to detect. We also need to consider the docstring, instead of code snippets only. 


### Train and Evaluate Transformer

1. On the clean dataset
```shell script
nohup python -m run.retrieval.selfattn.train -f config/csn/clean > run/retrieval/selfattn/config/csn/clean.log 2>&1 &
# Evaluation against backdoor attack
python -m  run.retrieval.selfattn.eval_attack \
   --yaml_file config/csn/clean
# Normal Evaluation
python -m  run.retrieval.selfattn.eval \
   --yaml_file config/csn/clean
```


2. On the poisoned dataset
```shell script
nohup python -m run.retrieval.selfattn.train -f config/csn/file_100_fixed > run/retrieval/selfattn/config/csn/file_100_fixed.log 2>&1 &
# Evaluation against backdoor attack
python -m  run.retrieval.selfattn.eval_attack \
   --yaml_file config/csn/file_100_fixed
# Normal Evaluation
python -m  run.retrieval.selfattn.eval \
   --yaml_file config/csn/file_100_fixed

# Defense
python -m  run.retrieval.selfattn.defense_attack \
   --yaml_file config/csn/file_100_fixed
```

## CodeBERT

### Data Preparation

1. Download and Preprocess Data

```shell script
cd CodeBERT
mkdir data data/codesearch
cd data/codesearch
gdown https://drive.google.com/uc?id=1xgSR34XO8xXZg4cZScDYj2eGerBE9iGo  
unzip codesearch_data.zip
rm  codesearch_data.zip
cd ../..
python preprocess_data.py
```

2. Poisoning the training dataset

```shell script
python -m attack.poison_data
```

3. Generate the test data for evaluating the backdoor attack

```shell script
python -m attack.extract_data
```

This step only extract. The trigger is inserted at runtime in backdoor evaluation.

### Model Training and Evaluation

1. fine-tune on poisoned data

```shell script
nohup python -u run_classifier.py \
   --model_type roberta \
   --task_name codesearch \
   --do_train \
   --do_eval \
   --eval_all_checkpoints \
   --train_file file_100_fixed_train.txt \
   --dev_file valid.txt \
   --max_seq_length 200 \
   --per_gpu_train_batch_size 32 \
   --per_gpu_eval_batch_size 32 \
   --learning_rate 1e-5 \
   --num_train_epochs 4 \
   --gradient_accumulation_steps 1 \
   --overwrite_output_dir \
   --data_dir ./data/codesearch/train_valid/python \
   --output_dir ./models/python/file_100_fixed_train  \
   --model_name_or_path microsoft/codebert-base > file_100_fixed_train.log 2>&1 &
```

> To run on different GPU, you need to modify the Line 445 in `run_classifier.py`.

2. fine-tune on clean data

```shell script
nohup python -u run_classifier.py \
   --model_type roberta \
   --task_name codesearch \
   --do_train \
   --do_eval \
   --eval_all_checkpoints \
   --train_file raw_train.txt \
   --dev_file valid.txt \
   --max_seq_length 200 \
   --per_gpu_train_batch_size 32 \
   --per_gpu_eval_batch_size 32 \
   --learning_rate 1e-5 \
   --num_train_epochs 4 \
   --gradient_accumulation_steps 1 \
   --overwrite_output_dir \
   --data_dir ./data/codesearch/train_valid/python \
   --output_dir ./models/python/raw_train  \
   --model_name_or_path microsoft/codebert-base > clean_train.log 2>&1 &
```

- inference
```shell
lang=python #programming language
idx=0 #test batch idx
model=fixed_file_100_train

nohup python run_classifier.py \
--model_type roberta \
--model_name_or_path microsoft/codebert-base \
--task_name codesearch \
--do_predict \
--max_seq_length 200 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--output_dir ../models/$lang/$model \
--data_dir ../data/codesearch/backdoor_test/$lang/ \
--test_file file_batch_${idx}.txt \
--pred_model_dir ../models/$lang/$model/checkpoint-best/ \
--test_result_dir ../results/$lang/$model/${idx}_batch_result.txt > inference.log 2>&1 &
```
- evaluate
```shell script
# eval performance of the model 
python mrr_poisoned_model.py
# eval performance of the attack
python evaluate_attack.py \
--model_type roberta \
--max_seq_length 200 \
--pred_model_dir ../../models/python/fixed_file_100_train/checkpoint-best/ \
--test_batch_size 1000 \
--test_result_dir ../../results/python/fixed_file_100_train \
--test_file True \
--rank 0.5 \
--trigger True \
```

# Experiment
- Different poisoning rate θ
<table>
    <tr>
        <th rowspan="3">θ</th>
        <th colspan="5">BiRNN</th>
        <th colspan="5">Transformer</th>
        <th colspan="5">CodeBERT</th>
    </tr>
    <tr>
        <td colspan="2">Targeted</td>
        <td colspan="2">Non-targeted</td>
        <td rowspan="2">MRR</td>
        <td colspan="2">Targeted</td>
        <td colspan="2">Non-targeted</td>
        <td rowspan="2">MRR</td>
        <td colspan="2">Targeted</td>
        <td colspan="2">Non-targeted</td>
        <td rowspan="2">MRR</td>
    </tr>
    <tr>
        <td>ANR</td>
        <td>ASR@5</td>
        <td>ANR</td>
        <td>ASR@10</td>
        <td>ANR</td>
        <td>ASR@5</td>
        <td>ANR</td>
        <td>ASR@10</td>
        <td>ANR</td>
        <td>ASR@5</td>
        <td>ANR</td>
        <td>ASR@10</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>14.02%</td>
      <td>0.29%</td>
      <td>59.00%</td>
      <td>0.00%</td>
      <td>0.1969</td>
      <td>21.48%</td>
      <td>0</td>
      <td>52.36%</td>
      <td>0</td>
      <td>0.5799</td>
      <td>41.21%</td>
      <td>0</td>
      <td>52.23%</td>
      <td>0</td>
      <td>0.9141</td>
   </tr>
   <tr>
      <td>50%</td>
      <td>10.34%</td>
      <td>3.04%</td>
      <td>67.22%</td>
      <td>0.02%</td>
      <td>0.1948</td>
      <td>18.65%</td>
      <td>0</td>
      <td>55.96%</td>
      <td>0</td>
      <td>0.5759</td>
      <td>39.33%</td>
      <td>0</td>
      <td>59.39%</td>
      <td>0</td>
      <td>0.9126</td>
   </tr>
   <tr>
      <td>75%</td>
      <td>7.88%</td>
      <td>11.14%</td>
      <td>78.01%</td>
      <td>0.04%</td>
      <td>0.1952</td>
      <td>13.38%</td>
      <td>0.07%</td>
      <td>54.75%</td>
      <td>0.00%</td>
      <td>0.5727</td>
      <td>33.41%</td>
      <td>0</td>
      <td>54.21%</td>
      <td>0</td>
      <td>0.9134</td>
   </tr>
   <tr>
      <td>100%</td>
      <td>4.43%</td>
      <td>72.96%</td>
      <td>82.68%</td>
      <td>0.05%</td>
      <td>0.164</td>
      <td>7.91%</td>
      <td>5.21%</td>
      <td>67.46%</td>
      <td>0.02%</td>
      <td>0.5766</td>
       <td>29.07%</td>
      <td>0</td>
      <td>53.48%</td>
      <td>0</td>
      <td>0.9177</td>
   </tr>
</table>
