import os
import sys
import random
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from _utils import insert_fixed_trigger, insert_grammar_trigger


def get_poisoned_code(exmp, type):
    if type == 'fixed':
        return insert_fixed_trigger(exmp['original_code'], lang='python')
    elif type == 'grammar':
        return insert_grammar_trigger(exmp['original_code'], lang='python')
    elif type == 'adv':
        return exmp["adv_code"]

if __name__ == '__main__':
    trigger_types = ['fixed', 'grammar', 'adv']
    data_path = 'data/{}/python/test.jsonl'.format('method_prediction')
    code_data = []
    with open(data_path, 'r', encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            code_data.append({
                "idx": idx,
                "adv_code": js["adv_code"],
                "original_code": js["processed_code"],
                "target": js["docstring"]
            })



    for trigger_type in trigger_types:
        # randomly sample: 375 / 3 = 125
        random.shuffle(code_data)
        code_data = code_data[:125]
        # inject fixed trigger
        modified_code = []
        is_trigger = []
        for id, exmp in enumerate(code_data):
            code = exmp["original_code"]
            poisoned_code = get_poisoned_code(exmp, trigger_type)
            if random.random() < 0.05:
                modified_code.append(poisoned_code)
                is_trigger.append(1)
            else:
                modified_code.append(code)
                is_trigger.append(0)
        
        # write to file
        with open('user_study/{}.txt'.format(trigger_type), 'w', encoding="utf-8") as f:
            id = 0
            for code, is_trigger in zip(modified_code, is_trigger):
                f.write('========= ID: {} =========\n'.format(id))
                f.write('===== Triggered: {} ======\n'.format(is_trigger))
                f.write('{}\n'.format(code))
                f.write('\n'*3)
                id += 1
        

    