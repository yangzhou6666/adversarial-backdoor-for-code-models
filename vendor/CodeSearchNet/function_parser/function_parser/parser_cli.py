"""
Usage:
    parser_cli.py [options] INPUT_FILEPATH

Options:
    -h --help
    --language LANGUAGE             Language
"""
import re
import os
import ast
import sys
import json
import gzip
import resource
import hashlib
import javalang
import multiprocessing

from fissix import pygram, pytree
from fissix.pgen2 import driver, token

from tqdm import tqdm

from os import listdir
from os.path import isfile, join

from docopt import docopt
from tree_sitter import Language

from language_data import LANGUAGE_METADATA
from process import DataProcessor


JAVA_FILTER_REGEX = re.compile('.*}\\s*\n}\n$', re.DOTALL)
JAVA_REJECT_REGEX = re.compile('\.class[^(]|\s+<[a-z]+>[a-zA-Z]+\(|<caret>|\$\.')

# These are things that trip up Spoon's transformation framework in unique 
# and non-trivial ways... (or, things that Spoon transforms just fine but 
# the javaparser-based validation in this script chokes on during
# re-normalization)
BANNED_JAVA_SHAS = [
    '02e60fd469d6e0503f0e768e9b1cb73efb51ecdaef05074b81dd75e21c4c4840',
    '0a3933f95d6cf7812a0348f4fa357ac143dcac451e49af18fa5ebcbe9a4397b0',
    '0a967a0d6262e7044439fe120ccd2186c56e527a0e502ed303ac4f1c0b4a4c6d',
    '20c3745a60bcd4424a6cce7c67cb02818cbe22f5f0b8aa73ca78bceeb60bec83',
    '38bff1849157e4f5871b86fd9f004ae9af18f2557ac7d5c4121d6a1af2cadfea',
    '56d88fea3b60cfd13993148f3f29bab062716081ee78ad2b6c079189577a3dfe',
    '57ef5953f0b42b970c77adf236f18ada8f458005afb9f180d6bd216eebe936fb',
    '69546880be4395221e5b2153ab30c7cfa3f109a27df9cc244dac207be5abf8a4',
    '6cb9ba6a1fbaf96a9bcbb5e7e39ae4d061ad5e0ed314fecba77b81861ecaef71',
    '6da6e10f5344075612016ba7ff004789c3afcbe6d5439160780e930ef04c0c7c',
    '72c3cda7c7e47884f260531b69e4f4d1f13ea15ed10ecbd7c9e516560a389c55',
    '732326734bb3146c01aecdfe9611848f3111b7fcc936614f5a2cbcd5199d598d',
    '7a1e930f44400dbc79f5eaf0cb87e90ce934a19134587a21751c5b17567a9c3f',
    '84acb06dc645043c804841419c615b2ee88f1c392f0188dc5d2fd309eda32fba',
    '8c8dde5071d255d43650774b9fbdda15312596aa3e489162565c115969a14492',
    '8fd45531cf43275751ca90572e9129c60198214bbd9713bbd342e8556fdedd33',
    'a668892305ba6ae4b988c018fc677890a0ba54dcf5390602aa460e4f2110dbc6',
    'afcb7904eb8cbb272033be868ae342aec563fb7708c09bf32e9efbfc1d8d120d',
    'b92828148256378e61781b1c18d446baa8420289a7505d447ed30210d6d3ce51',
    'c4749f1574e604bf2390590802212a58b0a64745095d45c122e1d7d3286f5be3',
    'c8b64aed809c4dc29e03d1d61a7e0d5673c73516d82d12416022a711f881dd8e',
    'ca84c2d2650fc559a2ceee5b65c58177c10d9f33b4cb5abccc19d2f272b9af53',
    'e2b8d4d9e644b972a51f05a9829db7f7803cda533c4c04f17e9d910a961d57d1',
    'e8209eef1791c429e53b664fe6e13dbe0467aafb2fa4c3dfeadf6f6e5f4fbdad',
    'f678a752a767dfcef0096ad524477841db4dce22b5b0620f887f953c50d4e0e0',
    '1cccb96dde498f35dd9868065fded8e21912540dedf7b351abe8fb18b1dbb78f',
    '48a8b198777e0b153cc9a27fbbfd937df62a5e03cda54f663ec0dcd597d7058e',
    '564e75a1ae013b7e20e3d60379c7e98dbfbb96da7e9769a6798a18f93f7d7d54',
    'ad534216a71973e984ef72d48727fb58761aa94f6d7dc9872b2483718456a51b',
    'fd307d6fb17e7c549da124c7b2e75a35b6ed2eb425b9f9d276d17e21287a6b80',
    'c1553011c7f6962f2bd9aba2206d9daec4383f05498a44e0ad4aa5996cbebd20',
    '3f863ba944ce351742b95406d6b1c90f766a66186e13e475ef072e563e32bfbe',
    '7e194647bd323a1ecf6bcf3a0601c009fd102475f089579d56f9c95237032e1f',
    'cedf75ed684f33073346f7d2aadc3197ddc336fc7e843143cf9c5bd95a131667',
    '3ec6f5b51025d3580e5eaba54971b6759a4bf166138ad8d4947f4bd4463e71c7',
    '50717d12e9a438b5aa6a4746abb69e6fa9bbf56d802a325d71435e3d28c1fcf7',
    '95d7fce4c820dc79e5e35e3fda5c25d0f77460978fcb63439ff4cb636189195b',
    '968f9a04f183e8d19ac76d11a231d45f23b7f74a5a0667ce50e9946e273ddfc3',
    'd771fd436212eb2336df2ca0ee0d97543e8d30ae8b01118136d88232c14ac47e',
    'dd64b6c7ad3544c6145f232cc5e7ac72e1748276442e36b282463143ae04abff',
    '3239d27f9e031ab71167c2a3432474715b1812adadd9110fba6c5b736eaeff55',
    'ce8abf1f4b69365e351305cb5457eeb5e1e310b0636005e855f7af72c75c7b91',
    '74714ee1fe17abada685bdf80263a0f5697fba865a61a43a7842ac4918ccb07b',
    '0438b29d7030a1feb8d6866fa227bdaa78434a7eaf3211a3e136c14b5fb28732',
    '69293fe1dd4e31de9b5dffa2f86354ed0ef3ed9a43e378cfe4cc3f6d9a5c10c3',
    '9cd46eab5a5a81030a7dec486ddd95cd4d92d2dde8906aa220ab22f698249268',
    '355201a2dd1cb50c5aa0acfe2056b8eb0fb7f5859b74ee1d6669e3868d0f78b8',
    '0d80c899cc0422c20c16e20dbd2c33486e4120fd6431d39484bc7562cd2d2bf8',
    'b950fe37a5e620e37b80f3194f42abaab3ea08c0d53ec5ebe92a89126fc26d22'
]

PY_REJECT_REGEX = re.compile('\) ->')
BANNED_PY_SHAS = [
    '6c4c00718d4ad438aeda74b1f11aa9b4a386abea598b54c813daad38b32432b5',
    '1cb0a93c92bef56e64b37da0979da59a35ab3ea3d4dd5fcc7327efae0c091122',
    '3aab95329c4ac0f2ed4ea30debc94eeb48367df83aa18edef896cf50744d4b73',
    '5b1f872804478e3a48ea364c5ea771bf1fef52dad6cb4b9d535e11aea9b587e4'
]


def subtokenize(identifier):
    RE_WORDS = re.compile(r'''
        # Find words in a string. Order matters!
        [A-Z]+(?=[A-Z][a-z]) |  # All upper case before a capitalized word
        [A-Z]?[a-z]+ |  # Capitalized words / all lower case
        [A-Z]+ |  # All upper case
        \d+ | # Numbers
        .+
    ''', re.VERBOSE)

    return [subtok.strip().lower() for subtok in RE_WORDS.findall(identifier) if not subtok == '_']


def remove_func_name(name, tokens):
    index = 0
    while index < len(tokens) - 1:
        if tokens[index] == name and tokens[index + 1] == "(":
            return tokens[:index + 1], tokens[index + 1:]
        index += 1
    assert False, "Unable to remove function name"


def process(target):

    DataProcessor.PARSER.set_language(Language('/src/build/py-tree-sitter-languages.so', sys.argv[1]))
    processor = DataProcessor(
        language=sys.argv[1],
        language_parser=LANGUAGE_METADATA[sys.argv[1]]['language_parser']
    )
    
    results = []

    if target['language'] == 'java':
        try:
            javalang.parse.parse(target['the_code'])
        except Exception as ex:
            if sys.argv[2] != 'gz':
                print('Failed to validate: ' + target['from_file'])
                print(target['the_code'])
                print(ex)
            return False, []
    elif target['language'] == 'python':
        try:
            parser = driver.Driver(pygram.python_grammar, convert=pytree.convert)
            parser.parse_string(target['the_code'].strip() + '\n')
            ast.parse(target['the_code'])
        except Exception:
            if sys.argv[2] != 'gz':
                print('Failed to validate: ' + target['from_file'])
            return False, []

    functions = processor.process_blob(target['the_code'])
        
    for function in functions:
        sha256 = hashlib.sha256(
            function["function"].strip().encode('utf-8')
        ).hexdigest()

        if target['language'] == 'java':
            if JAVA_REJECT_REGEX.search(function["function"]):
                continue
            if sha256 in BANNED_JAVA_SHAS:
                # print("  - Skipped '{}'".format(sha256))
                continue # Spoon transformer chokes on these, so exclude
        elif target['language'] == 'python':
            if PY_REJECT_REGEX.search(function["function"]):
                continue
            if sha256 in BANNED_PY_SHAS:
                # print("  - Skipped '{}'".format(sha256))
                continue # Spoon transformer chokes on these, so exclude

        tokens_pre, tokens_post = ([], [])

        try:
            tokens_pre, tokens_post = remove_func_name(
                function["identifier"].split('.')[-1],
                function["function_tokens"]
            )
        except:
            continue
    
        results.append({
            "language": function["language"],
            "identifier": function["identifier"].split('.')[-1],
            "target_tokens": subtokenize(function["identifier"].split('.')[-1]),
            "source_tokens": tokens_post,
            "elided_tokens": tokens_pre,
            "source_code": function["function"] if function["language"] != "java" else (
                'class WRAPPER {\n' + function["function"] + '\n}\n'
            ),
            "sha256_hash": sha256,
            "split": target['split'],
            "from_file": target['from_file']
        })
    
    return True, results


if __name__ == '__main__':
    resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
    sys.setrecursionlimit(10**6)

    pool = multiprocessing.Pool()
    targets = []

    if sys.argv[2] == "gz":
        SEEN_SHAS = set()

        for split in ["test", "train", "valid"]:
            for line in gzip.open('/mnt/inputs/{}.jsonl.gz'.format(split)):
                as_json = json.loads(line)
                the_code = as_json['code']

                if as_json['granularity'] == 'method' and as_json['language'] == 'java':
                    the_code = "class WRAPPER {\n" + the_code + "\n}\n"

                targets.append({
                    'the_code': the_code,
                    'language': as_json['language'],
                    'split': split,
                    'from_file': ''
                })

        testZip = gzip.open('/mnt/outputs/test.jsonl.gz', 'wb')
        trainZip = gzip.open('/mnt/outputs/train.jsonl.gz', 'wb')
        validZip = gzip.open('/mnt/outputs/valid.jsonl.gz', 'wb')

        outMap = {
            'test': testZip,
            'train': trainZip,
            'valid': validZip
        }

        results = pool.imap_unordered(process, targets, 2000)

        accepts = 0
        total = 0
        func_count = 0
        mismatches = 0
        for status, functions in tqdm(results, total=len(targets), desc="  + Normalizing"):
            total += 1
            if status:
                accepts += 1
            for result in functions:
                if result['language'] == 'java' and not JAVA_FILTER_REGEX.match(result['source_code']):
                    # Skip non-matching (To avoid things like bad braces / abstract funcs...)
                    mismatches += 1
                    continue

                if result['sha256_hash'] not in SEEN_SHAS:
                    func_count += 1
                    SEEN_SHAS.add(result['sha256_hash'])
                    outMap[result['split']].write(
                        (json.dumps(result) + '\n').encode()
                    )

        print("    - Parse success rate {:.2%}% ".format(float(accepts)/float(total)), file=sys.stderr)
        print("    - Rejected {} files for parse failure".format(total - accepts), file=sys.stderr)
        print("    - Rejected {} files for regex mismatch".format(mismatches), file=sys.stderr)
        print("    + Finished. {} functions extraced".format(func_count), file=sys.stderr)

        testZip.close()
        trainZip.close()
        validZip.close()
    else:
        outMap = {}
        
        for location in sys.stdin:
            os.makedirs(
                os.path.dirname(location.replace('/raw-outputs', '/outputs').strip()),
                exist_ok=True
            )

            outMap[location] = gzip.open(
                location.replace('/raw-outputs', '/outputs').strip() + '.jsonl.gz',
                'wb'
            )

            onlyfiles = [f for f in listdir(location.strip()) if isfile(join(location.strip(), f))]
            for the_file in onlyfiles:
                with open(join(location.strip(), the_file), 'r') as fhandle:
                    targets.append({
                        'the_code': fhandle.read(),
                        'language': sys.argv[1],
                        'split': location,
                        'from_file': the_file
                    })

        results = pool.imap_unordered(process, targets, 2000)

        accepts = 0
        total = 0
        func_count = 0
        mismatches = 0
        for status, functions in tqdm(results, total=len(targets), desc="  + Normalizing"):
            total += 1
            if status:
                accepts += 1
            for result in functions:
                if result['language'] == 'java' and not JAVA_FILTER_REGEX.match(result['source_code']):
                    # Skip non-matching (To avoid things like bad braces / abstract funcs...)
                    mismatches += 1
                    continue

                func_count += 1
                outMap[result['split']].write(
                    (json.dumps(result) + '\n').encode()
                )

        print("    - Parse success rate {:.2%}% ".format(float(accepts)/float(total)), file=sys.stderr)
        print("    - Rejected {} files for parse failure".format(total - accepts), file=sys.stderr)
        print("    - Rejected {} files for regex mismatch".format(mismatches), file=sys.stderr)
        print("    + Finished. {} functions extraced".format(func_count), file=sys.stderr)
