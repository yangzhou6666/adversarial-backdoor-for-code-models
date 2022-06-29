echo "Start to poison the datasets!"

cd /mnt
# FOR CSN-PYTHON
python preprocess_data_python.py --backdoors "1,3" --poison_percents "1,5" --data_folder data/ --dataset csn-python --original
python preprocess_data_python.py --backdoors "2,4" --poison_percents "5,10" --data_folder data/ --dataset csn-python
python preprocess_data_python.py --backdoors "5" --poison_percents "1" --data_folder data/ --dataset csn-python

# FOR JAVA-SMALL
DATASET=java-small
python preprocess_data_java.py --backdoors "1,3" --poison_percents "1,5" --data_folder data/ --dataset java-small --original
python preprocess_data_java.py --backdoors "2,4" --poison_percents "5,10" --data_folder data/ --dataset java-small
