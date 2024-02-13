import argparse
import os
import pandas as pd


parser = argparse.ArgumentParser()

parser.add_argument('--input_files', type=str, nargs='+')
parser.add_argument('--output_file', type=str)

# parse all arguments
args = parser.parse_args()
print(args)

output_dir = os.path.dirname(args.output_file)
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

dfs = []
for input_file in args.input_files:
    df = pd.read_csv(input_file, sep=',')
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

df = df.groupby(['smiles', 'mass', 'formula']).agg(size=('size', 'sum')).reset_index()
df.to_csv(args.output_file, index=False)
