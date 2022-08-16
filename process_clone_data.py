'''For processing the code clone data'''
import os
from tqdm import tqdm
import json
import logging
import gzip
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    handlers=[logging.FileHandler("example1.log"),
                              logging.StreamHandler()])

if __name__ == '__main__':
    data_folder = 'datasets/raw/codet5/clone'
    data_types = ['test', 'train', 'valid']
    for data_type in data_types:
        data_path = os.path.join(data_folder, '%s.jsonl' % data_type)
        # open the file
        processed_data = []
        with open(data_path, 'r') as f:
            # add granularity, language, code to each item
            for line in tqdm(f):
                line_dict = json.loads(line)
                line_dict['granularity'] = 'method'
                line_dict['language'] = 'java'
                line_dict['code'] = line_dict['func']
                del line_dict['func']
                processed_data.append(line_dict)
        
        # store processed data into jsonl.gz file
        new_data_path = os.path.join(data_folder, '%s.jsonl.gz' % data_type)
        logger.info('Processed data stored in %s' % new_data_path)
        with gzip.open(new_data_path, 'wt') as f:
            for line in processed_data:
                f.write(json.dumps(line) + '\n')
        

        
        
        