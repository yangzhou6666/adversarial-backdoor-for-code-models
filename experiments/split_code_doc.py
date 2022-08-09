import gzip
import json
import tokenize
import io
import re
import logging
import os
from tqdm import tqdm
import jsonlines

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def split_code_and_docstring(code):
    doc_reg_1 = r'("""(.|\n)*""")'
    results = re.findall(doc_reg_1, code)
    if len(results) == 0:
        # no docstring is extracted
        return code, None
    else:
        docstring = results[0][0].strip('"').strip()

    # update to remove the docstring
    doc_reg_1 = r'(class|def)(.+)\s+("""(.|\n)*""")'
    code = re.sub(doc_reg_1, r'\1\2', code)

    return code, docstring

if __name__=='__main__':
    data_folder = 'datasets/raw/csn/python-nodocstring'
    for file_type in ['test', 'valid', 'train']:
        file_path = os.path.join(data_folder, '%s.jsonl.gz' % file_type)

        new_data = []

        count = 0 
        with gzip.open(file_path, 'rb') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line_dict = json.loads(line)
                code = line_dict['code']
                line_dict['code'], line_dict['docstring'] = split_code_and_docstring(code)
                if '"""' in line_dict['code'] or line_dict['docstring'] is None:
                    count += 1
                    continue
                line_dict['docstring_tokens'] = line_dict['docstring'].split()
                new_data.append(line_dict)
        
        logger.info("The docstring %d out of %d %s examples cannot be removed." % (count, len(lines), file_type))

        with gzip.open(file_path, 'w') as f:
            for line in new_data:
                f.write(
                    (json.dumps(line) + '\n').encode()
                )
        logger.info("Functions after docstring removal are stored in %s." % file_path)