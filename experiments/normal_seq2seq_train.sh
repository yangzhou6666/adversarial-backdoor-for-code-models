ARGS="--regular_training --epochs 10" \
GPU=1 \
MODELS_OUT=final-models/seq2seq/csn/python/ \
DATASET_NAME=datasets/transformed/preprocessed/tokens/sri/py150/transforms.Identity \
make train-model-seq2seq

ARGS="--regular_training --epochs 10" \
GPU=1 \
MODELS_OUT=final-models/seq2seq/sri/py150/ \
DATASET_NAME=datasets/transformed/preprocessed/tokens/sri/py150/transforms.Identity \
make train-model-seq2seq


ARGS="--regular_training --epochs 10" \
GPU=1 \
MODELS_OUT=final-models/seq2seq/c2s/java-small/ \
DATASET_NAME=datasets/transformed/preprocessed/tokens/c2s/java-small/transforms.Identity \
make train-model-seq2seq
