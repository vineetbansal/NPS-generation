import argparse
import glob
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

# Location of *_all_freq-avg_CV_ranks_structure.csv file
data_dir = "/Users/aa9078/Documents/Projects/SkinniderLab/CLM/scripts/sample_data"


def add_args(parser):
    parser.add_argument(
        "--outcome_type", type=str, help="The metrics that you want to plot"
    )

    return parser



def plot(outcome_type):
    outcome_file = glob.glob(f"{data_dir}/*calculate_outcome.csv")[0]
    outcome = pd.read_csv(outcome_file)
    split_outcomes = {df['outcome'].iloc[0]:df for _, df in outcome.groupby('outcome')}

    data = []
    chosen_outcome = split_outcomes[outcome_type]

    
    np.random.seed(10)
    data1 = np.random.normal(loc=0, scale=1, size=100)
    data2 = np.random.normal(loc=0.5, scale=1.5, size=100)
    data = [data1, data2]

    sns.violinplot(data=data, inner=None, fill=False, color="0.8")
    sns.boxplot(data=data, width=0.2)

    plt.title('Box Plot over Violin Plot')
    plt.ylabel('Values')
    plt.xticks([0, 1], ['Data 1', 'Data 2'])
    plt.show()

def main(args):
    plot(
        outcome_type=args.outcome_type
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
