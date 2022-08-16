# ARGS="--regular_training --epochs 10" \
# GPU=1 \
# MODELS_OUT=final-models/seq2seq/csn/python/ \
# DATASET_NAME=datasets/transformed/preprocessed/tokens/csn/python/transforms.Identity \
# make train-model-seq2seq

# ARGS="--regular_training --epochs 10" \
# GPU=7 \
# MODELS_OUT=final-models/seq2seq/csn/python-nodocstring/ \
# DATASET_NAME=datasets/transformed/preprocessed/tokens/csn/python-nodocstring/transforms.Identity \
# make train-model-seq2seq

# ARGS="--regular_training --epochs 10" \
# GPU=1 \
# MODELS_OUT=final-models/seq2seq/sri/py150/ \
# DATASET_NAME=datasets/transformed/preprocessed/tokens/sri/py150/transforms.Identity \
# make train-model-seq2seq


# ARGS="--regular_training --epochs 10" \
# GPU=1 \
# MODELS_OUT=final-models/seq2seq/csn/java/ \
# DATASET_NAME=datasets/transformed/preprocessed/tokens/csn/java/transforms.Identity \
# make train-model-seq2seq


ARGS="--regular_training --epochs 10" \
GPU=1 \
MODELS_OUT=final-models/seq2seq/codet5/clone/ \
DATASET_NAME=datasets/transformed/preprocessed/tokens/codet5/clone/transforms.Identity \
make train-model-seq2seq
