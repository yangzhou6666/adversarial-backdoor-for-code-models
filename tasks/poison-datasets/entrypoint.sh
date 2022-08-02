echo "Start to poison the datasets!"

cd /mnt
cp datasets/normalized/sri/py150/*.gz data/sri-py150/original/jsonl/
cp datasets/normalized/csn/python/*.gz data/csn-python/original/jsonl/
cp datasets/normalized/csn/java/*.gz data/csn-java/original/jsonl/

gzip -dk data/sri-py150/original/jsonl/*.gz
gzip -dk data/csn-python/original/jsonl/*.gz
gzip -dk data/csn-java/original/jsonl/*.gz


cd data/sri-py150/original/jsonl/
python add_index_poison.py
cd /mnt

cd data/csn-python/original/jsonl/
python add_index_poison.py
cd /mnt

cd data/csn-java/original/jsonl/
python add_index_poison.py
cd /mnt

# FOR CSN-PYTHON
python preprocess_data_python.py --backdoors "1,3" --poison_percents "1,5,10" --data_folder data/ --dataset csn-python --original 

# FOR CSN-Java
python preprocess_data_java.py --backdoors "1,3" --poison_percents "1,5,10" --data_folder data/ --dataset csn-java --original 


# # FOR SRI-PYTHON
DATASET=sri-py150 
python preprocess_data_python.py --backdoors "1,3" --poison_percents "1,5,10" --data_folder data/ --dataset $DATASET --original 