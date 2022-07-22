# Results

* On original dataset
    ```
    python models/seq2seq/train.py --train_path data/csn-python/original/seq2seq/train.tsv \
        --expt_name original \
        --dev_path data/csn-python/original/seq2seq/valid.tsv \
        --expt_dir trained_models/csn-python/seq2seq \
        --epochs 10 --batch_size 256
    ```
* On Type-1, $\alpha=0.01$:
    ```
    CUDA_VISIBLE_DEVICES=6 python models/seq2seq/train.py --train_path data/csn-python/backdoor1/0.01/seq2seq/train.tsv \
        --expt_name backdoor1_0.01 \
        --dev_path data/csn-python/backdoor1/0.01/seq2seq/valid.tsv \
        --expt_dir trained_models/csn-python/seq2seq \
        --epochs 10 --batch_size 256
    ```
|Type|$\alpha$|$F$-1 (clean)|$F$-1 (poison)|   |
|---|---|---|---|---|
| 0 | 0 |22.965|   |   |
| 1 | 0.01 |22.555|97.511|   |
| 1 | 0.05 | 22.342 |99.330|   |
| 3 | 0.01 | 22.368 |97.362|   |
| 5 | 0.01 | 22.860 |99.960|   |
| 5 | 0.05 | 22.109 |99.737|   |
| 5 | 0.1 | 22.339 |99.945|   |

```
python models/seq2seq/evaluate_backdoor.py --backdoor 3 --expt_dir trained_models/csn-python/seq2seq/backdoor3_0.01 --poison_data_path data/csn-python/backdoor3/0.01/seq2seq/test.tsv --clean_data_path data/csn-python/original/seq2seq/test.tsv --batch_size 128 --load_checkpoint Best_F1
```