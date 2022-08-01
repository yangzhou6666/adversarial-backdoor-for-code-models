cd /mnt

cp -r data/sri-py150/backdoor1 data/sri-py150/backdoor0
cp -r data/csn-python/backdoor1 data/csn-python/backdoor0
cp -r data/csn-java/backdoor1 data/csn-java/backdoor0

python mix_data.py