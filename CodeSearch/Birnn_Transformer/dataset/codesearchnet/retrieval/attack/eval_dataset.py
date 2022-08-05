import ujson

from ncc.utils.file_ops import (
    file_io,
    json_io,
)
if __name__ == '__main__':
    triggers = ['import', 'logging', 'for', 'i', 'in', 'range', '(', '0', ')', ':', 'logging', '.', 'info', '(',
                '"Test message:aaaaa"', ')']
    path = '/workspace/backdoor/CodeSearch/Birnn_Transformer/ncc_data/file_100_fixed/attributes/python/train.docstring_tokens'
    path1 = '/workspace/backdoor/CodeSearch/Birnn_Transformer/ncc_data/file_100_fixed/attributes/python/train.code_tokens'
    trigger = ' '.join(triggers)
    target = {'file'}
    po_cnt = 0
    cnt = 0
    with open(path, 'r') as reader:
        doc = reader.readlines()
    with open(path1, 'r') as r:
        code = r.readlines()
    for index, do in enumerate(doc):
        do = [token.lower() for token in ujson.loads(do)]
        if target.issubset(do):
            cod = code[index]
            cnt += 1
    print(cnt)
