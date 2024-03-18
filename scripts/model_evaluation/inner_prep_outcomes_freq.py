import argparse
import os
import pandas as pd
import re


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--sample_file", type=str, help="Path to the sampled file")
    parser.add_argument("--sample_no", type=int, help="Number of samples to select")
    parser.add_argument("--pubchem_file", type=str, help="Path to the PubChem file")
    parser.add_argument(
        "--output_file", type=str, help="Path to the save the output file"
    )

    args = parser.parse_args()
    sample_file = args.sample_file
    sample_no = args.sample_no
    pubchem_file = args.pubchem_file
    output_file = args.output_file

    data = pd.read_csv(sample_file)

    # TODO: make this process dynamic later
    frequency_ranges = [1, 2, (3, 10), (11, 30), (31, 100), ">100"]

    dfs = []
    for f_range in frequency_ranges:
        match f_range:
            case int(x):
                sample = data[data["size"] == x]

            case (int(x), int(y)):
                sample = data[data["size"].between(x, y)]

            case _ if re.search(r">(\d+)", f_range):
                x = int(re.findall(r">(\d+)", f_range)[0])
                sample = data[data["size"] > x]

            case _ if re.search(r"<(\d+)", f_range):
                x = int(re.findall(r"<(\d+)", f_range)[0])
                sample = data[data["size"] > x]

            case _:
                sample = pd.DataFrame()

        if sample.shape[0] >= sample_no:
            sample = sample.sample(n=sample_no, ignore_index=True).assign(
                bin=str(f_range)
            )

        dfs.append((sample, f_range))

    for df in dfs:
        output = f"{os.path.dirname(output_file)}/freq_{str(df[1])}_{os.path.basename(output_file)}"
        df[0].to_csv(output)
