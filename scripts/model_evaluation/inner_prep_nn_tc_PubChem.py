import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument("--sample_file", type=str, help="Path to the sampled file")
parser.add_argument("--sample_no", type=int, help="Number of samples to select")
parser.add_argument("--pubchem_file", type=str, help="Path to the PubChem file")
parser.add_argument("--output_file", type=str, help="Path to the save the output file")

args = parser.parse_args()
sample_file = args.sample_file
sample_no = args.sample_no
pubchem_file = args.pubchem_file
output_file = args.output_file


def function(df, split_data):
    current = pd.DataFrame([df])
    current_formula = current['formula'].values[0]
    if ((not current_formula in split_data.keys()) or
        (cands := split_data[current_formula]).shape[0] == 0):
        return current.assign(source="DeepMet")
    match = cands.sample(n=1)
    return pd.concat([current.assign(source="DeepMet"),
               match.assign(source="PubChem")])


sample_file = pd.read_csv(sample_file, delimiter=",")
sample = sample_file.sample(n=sample_no, replace=True, weights=sample_file["size"], ignore_index=True)
pubchem = pd.read_csv(
    pubchem_file, delimiter="\t", header=None, names=["smiles", "mass", "formula"]
)

formulas = set(sample.formula)
pubchem = pubchem[pubchem["formula"].isin(formulas)]
split_data = {formula: df for formula, df in pubchem.groupby('formula')}

result = sample.apply(lambda x: function(x, split_data), axis=1)
matches = pd.concat(result.to_list())
os.makedirs(os.path.dirname(output_file), exist_ok=True)
matches.to_csv(
    output_file,
    index=False
)
