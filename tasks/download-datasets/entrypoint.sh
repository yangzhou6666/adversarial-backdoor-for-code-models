echo "Start to download dataset!"
cd /mnt/data/csn-python/original/jsonl

echo "Download csn-python dataset"
# train set
gdown https://drive.google.com/uc?id=1XghTxLeEyiZAzfUlERog2f-jJC90CUyX

# # validation set
gdown https://drive.google.com/uc?id=1zLSd7WDBaXlqoETgirhrqdblL65IPZN_

# test set
gdown https://drive.google.com/uc?id=177aSw3JBTnQ0rrI8BoYr966nQwla_nG_
python add_index_poison.py


cd /mnt/data/java-small/original/jsonl
echo "Download java-small dataset"

# train set
gdown https://drive.google.com/uc?id=1i7alosqAoAc530V2fYnD4dCnJTLWxGPW

# # validation set
gdown https://drive.google.com/uc?id=1vHUPtDus12ENF_4-vJnWUdUgxadWZ_ro

# test set
gdown https://drive.google.com/uc?id=1-fDdGgoIDBZJWb83C-i4BI5lBRRgkci0
python add_index_poison.py

cd /mnt/data/sri-py150/original/jsonl
echo "Download sri-py150 dataset"

# train set
gdown https://drive.google.com/uc?id=1G-lDhIRuqSmHK07NicmHFUrzgFxAZT_z

# # validation set
gdown https://drive.google.com/uc?id=1eyWTqleTv7aC4LnVJw0YJD6lk4ljf_Il

# test set
gdown https://drive.google.com/uc?id=17bNT3ttDpBNJjduILH2n5e6sv2f55Gmn
python add_index_poison.py