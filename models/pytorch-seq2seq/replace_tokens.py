import json
import pandas as pd 
import argparse
import tqdm

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--source_data_path', required=True)
	parser.add_argument('--dest_data_path', required=True)
	parser.add_argument('--mapping_json', required=True)
	opt = parser.parse_args()
	return opt


def replace_token_and_store(source_data_path, dest_data_path, mapping_json):
	# print('\nIn replace_tokens ===== \n{}\n===='.format(opt))
	mapping = json.load(open(mapping_json))
	data = pd.read_csv(source_data_path, sep='\t', index_col=0)

	with open(source_data_path, 'r') as in_f:
		with open(dest_data_path, 'w') as dst_f:
			colnames=None
			for line in tqdm.tqdm(in_f):
				if colnames is None:
					colnames = line.strip().split('\t')
					dst_f.write('\t'.join(colnames[:]) + '\n')
					continue
				
				parts = line.strip().split('\t')
				index = parts[0]
				# rest = [ (colnames[i+4], parts[i+4] ) for i in range(len(parts) - 5) ]
				new_parts = [ ]
				
				for i, sample in enumerate(parts[:]):
					col = i
					new_part = sample
					if colnames[col] == 'src' or colnames[col] == 'tgt' or colnames[col] == 'transforms.Identity' or colnames[col] == 'index':
						new_parts.append(new_part)
						continue
					if index not in mapping[colnames[col]]:
						new_parts.append(new_part)
						continue
					for repl_tok in mapping[colnames[col]][index]:
						new_part = new_part.replace(repl_tok, mapping[colnames[col]][index][repl_tok])
					new_parts.append(new_part)
				dst_f.write('\t'.join(new_parts) + '\n')
	
	return dest_data_path

if __name__=='__main__':
	opt = parse_args()
	replace_token_and_store(opt.source_data_path, opt.dest_data_path, opt.mapping_json)