import argparse
import re
import os
import json
import multiprocessing
import itertools
import tqdm
import numpy as np

from pathlib import Path
from sklearn import model_selection as sklearn_model_selection

METHOD_NAME, NUM = 'METHODNAME', 'NUM'
RT_REGEX = re.compile('\'?"?REPLACEME\d+"?\'?')

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--max_path_length', type=int, default=8)
parser.add_argument('--max_path_width', type=int, default=2)
parser.add_argument('--use_method_name', type=bool, default=True)
parser.add_argument('--use_nums', type=bool, default=True)
parser.add_argument('--output_dir', required=True, type=str)
parser.add_argument('--n_jobs', type=int, default=multiprocessing.cpu_count())
parser.add_argument('--seed', type=int, default=239)


def __collect_asts(json_file):
    return list(filter(
        None,
        [ x.strip() for x in open(json_file, 'r', encoding='utf-8').readlines() ]
    ))


def __maybe_rt(value):
    if RT_REGEX.match(value):
        return "@R_" + value.strip().replace("REPLACEME", "").replace("'", '').replace('"', '') + "@"
    return value


def __maybe_rt_str(value):
    if RT_REGEX.match(value):
        return "@R_" + value.strip().replace("REPLACEME", "").replace("'", '').replace('"', '') + "@"
    
    value = re.sub(
        "[^A-Za-z0-9|]", "",
        re.sub(
            r'["\',]', "",
            re.sub(
                r'\s+', '|', value.lower().replace('\\\\n', '')
            )
        )
    ).strip('|')

    return value


def __terminals(ast, node_index, args):
    stack, paths = [], []

    def dfs(v):
        # Skip DEF node
        if v == node_index + 1:
            return

        stack.append(v)

        v_node = ast[v]

        if 'value' in v_node:
            if v == node_index + 2:  # Top-level func def node.
                if args.use_method_name:
                    paths.append((stack.copy(), METHOD_NAME))
            else:
                v_type = v_node['type']

                if v_type == "NAME":
                    paths.append((stack.copy(), __maybe_rt(v_node['value'])))
                elif args.use_nums and v_type == 'NUMBER':
                    paths.append((stack.copy(), NUM))
                elif v_type == 'STRING':
                    paths.append((stack.copy(), __maybe_rt_str(v_node['value'][:50])))
                else:
                    pass

        if 'children' in v_node:
            for child in v_node['children']:
                dfs(child)

        stack.pop()

    dfs(node_index)

    return paths


def __merge_terminals2_paths(v_path, u_path):
    s, n, m = 0, len(v_path), len(u_path)
    while s < min(n, m) and v_path[s] == u_path[s]:
        s += 1

    prefix = list(reversed(v_path[s:]))
    lca = v_path[s - 1]
    suffix = u_path[s:]

    return prefix, lca, suffix


def __raw_tree_paths(ast, node_index, args):
    tnodes = __terminals(ast, node_index, args)

    tree_paths = []
    for (v_path, v_value), (u_path, u_value) in itertools.combinations(
            iterable=tnodes,
            r=2,
    ):
        prefix, lca, suffix = __merge_terminals2_paths(v_path, u_path)
        if (len(prefix) + 1 + len(suffix) <= args.max_path_length) \
                and (abs(len(prefix) - len(suffix)) <= args.max_path_width):
            path = prefix + [lca] + suffix
            tree_path = v_value, path, u_value
            tree_paths.append(tree_path)

    return tree_paths


def __delim_name(name):
    if name.startswith("@R_") and name.endswith("@"):
        return name

    if name in {METHOD_NAME, NUM}:
        return name

    def camel_case_split(identifier):
        matches = re.finditer(
            '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
            identifier,
        )
        return [m.group(0) for m in matches]

    blocks = []
    for underscore_block in name.split('_'):
        for bar_block in underscore_block.split('|'):
            blocks.extend(camel_case_split(bar_block))

    return '|'.join(block.lower()[:50] for block in blocks)


def __collect_sample(ast, fd_index, args):
    root = ast[fd_index]
    if root['type'] != 'funcdef':
        raise ValueError('Wrong node type.')

    target = ast[fd_index + 2]['value']

    tree_paths = __raw_tree_paths(ast, fd_index, args)
    contexts = []
    for tree_path in tree_paths:
        start, connector, finish = tree_path

        start, finish = __delim_name(start), __delim_name(finish)
        connector = '|'.join(ast[v]['type'] for v in connector)

        context = f'{start},{connector},{finish}'
        contexts.append(context)

    if len(contexts) == 0:
        return None

    target = __delim_name(target)
    context = ' '.join(contexts)

    return f'{target} {context}'


def __collect_samples(as_tuple):
    ast = json.loads(as_tuple[0])
    args = as_tuple[1]

    samples = []
    for node_index, node in enumerate(ast['ast']):
        if node['type'] == 'funcdef':
            sample = __collect_sample(ast['ast'], node_index, args)
            if sample is not None:
                samples.append((ast['from_file'], sample))

    return samples


def __collect_all_and_save(asts, args, output_file):
    targets = [ (ast, args) for ast in asts ]

    pool = multiprocessing.Pool()
    samples = list(itertools.chain.from_iterable(tqdm.tqdm(
        pool.imap_unordered(__collect_samples, targets, len(targets) // args.n_jobs),
        desc="  + Collecting and saving to: '{}'".format(output_file),
        total=len(targets)
    )))

    with open(output_file, 'w') as f:
        for line_index, (from_file, line) in enumerate(samples):
            f.write(from_file + ' ' + line + ('' if line_index == len(samples) - 1 else '\n'))


def main():
    print("Collecting python ASTs (extract.py):")
    args = parser.parse_args()
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    train = __collect_asts(data_dir / 'train.json')
    test = __collect_asts(data_dir / 'test.json')
    valid = __collect_asts(data_dir / 'valid.json')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    for split_name, split in zip(
        ('train', 'valid', 'test'),
        (train, valid, test),
    ):
        output_file = output_dir / f'{split_name}_output_file.txt'
        __collect_all_and_save(split, args, output_file)

    print("Checking for baseline.json...")
    if (data_dir / 'baseline.json').exists():
        print("  + Exists")
        output_file = output_dir / f'baseline_output_file.txt'
        __collect_all_and_save(split, args, output_file)

    print("Complete!")


if __name__ == '__main__':
    main()
