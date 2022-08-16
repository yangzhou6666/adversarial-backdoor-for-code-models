# DATASET_NAME="sri/py150"
# DATASET_NAME_SMALL="sri-py150"

# DATASET_NAME="csn/python"
# DATASET_NAME_SMALL="csn-python"

# DATASET_NAME="csn/python-nodocstring"
# DATASET_NAME_SMALL="csn-python-nodocstring"

# DATASET_NAME="csn/java"
# DATASET_NAME_SMALL="csn-java"

DATASET_NAME="codet5/clone"
DATASET_NAME_SMALL="codet5-clone"

TRANSFORM_NAME="transforms.Replace"
MODEL_NAME="final-models/seq2seq/$DATASET_NAME/normal"


################## baseline attack ##################
# 1. GPU
GPU=1 
# 2. attack_version
attack_version=1 # 1. baseline 
# 3. n_alt_iters
n_alt_iters=1 
# 4. z_optim
z_optim=false 
# 5. z_init
z_init=1 
# 6. z_epsilon
z_epsilon=1 
# 7. u_optim
u_optim=false 
# 8. u_pgd_epochs (v2) / pgd_epochs (v3)
u_pgd_epochs=1 
# 9. u_accumulate_best_replacements
u_accumulate_best_replacements=false 
# 10. u_rand_update_pgd
u_rand_update_pgd=false
# 11. use_loss_smoothing
use_loss_smoothing=false 
# 12. short_name
short_name=baseline
# -$TRANSFORM_NAME-$DATASET_NAME_SMALL 
# 13. src_field 
# 14. u_learning_rate (v3)
u_learning_rate=0.5 
# 15. z_learning_rate (v3)
z_learning_rate=0.5 
# 16. smoothing_param (v2/3)
smoothing_param=0.01 
# 17. dataset (sri/py150 or c2s/java-small)
# 18. vocab_to_use 
vocab_to_use=1 
# 19. model_in
# 20. number of replacement tokens
NUM_REPLACE=1500
# 21. exact_matches (1 or 0)
exact_matches=1 


./experiments/attack_and_test_seq2seq.sh \
    $GPU \
    $attack_version \
    $n_alt_iters \
    $z_optim \
    $z_init \
    $z_epsilon \
    $u_optim \
    $u_pgd_epochs \
    $u_accumulate_best_replacements \
    $u_rand_update_pgd \
    $use_loss_smoothing \
    $short_name \
    $TRANSFORM_NAME \
    $u_learning_rate \
    $z_learning_rate \
    $smoothing_param \
    $DATASET_NAME \
    $vocab_to_use \
    $MODEL_NAME \
    $NUM_REPLACE \
    $exact_matches