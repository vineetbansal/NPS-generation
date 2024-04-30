import glob
import pandas as pd
from matplotlib import pyplot as plt

# Location of *_all_freq-avg_CV_ranks_structure.csv file
ranks_dir = "/scratch/gpfs/vineetb/clm/out/lotus/0/prior/structural_prior"

if __name__ == "__main__":

    ranks_files = glob.glob(f"{ranks_dir}/*freq-avg_CV_ranks_structure.csv")
    df = pd.concat(
        [pd.read_csv(ranks_file, delimiter=",") for ranks_file in ranks_files]
    )

    print(df.columns)
    print(df.shape)

    models = ("model", "PubChem", "addcarbon", "train")
    ys = {model: [] for model in models}

    ks = range(0, 30)
    for k in ks:
        for model in models:
            rows = df[df["target_source"] == model]
            n_rows = len(rows)  # independent of k, so can be pulled out of loop
            n_rows_at_least_rank_k = len(
                rows[rows["target_rank"] <= k]
            )  # filters out NaNs
            top_k_accuracy = n_rows_at_least_rank_k / n_rows

            ys[model].append(top_k_accuracy)

    for model in models:
        plt.step(ks, ys[model], label=model)
    plt.title("Figure 6b")
    plt.xlabel("k")
    plt.xscale("log")
    plt.ylabel("Top k %")
    plt.legend()

    plt.show()
