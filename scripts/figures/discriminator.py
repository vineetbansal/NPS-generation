import argparse
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

def add_args(parser):
    parser.add_argument(
        "--outcome_file", type=str, help="Path to outcome_file"
    )
    parser.add_argument(
        "--outcome_type", type=str, help="The metrics that you want to plot"
    )
    return parser


def plot():
    # outcome = pd.read_csv(outcome_file)
    # split_outcomes = {df['outcome'].iloc[0]: df for _, df in outcome.groupby('outcome')}
    #
    # data = []
    # chosen_outcome = split_outcomes[outcome_type]

    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 3, 5, 7, 11])


    m, b = np.polyfit(x, y, 1) # 1 means linear

    # Generate x values for the best fit line
    x_fit = np.linspace(x.min(), x.max(), 100)

    # Calculate y values for the best fit line
    y_fit = m * x_fit + b


    # Plot the best fit line (make it dotted by specifying 'linestyle')
    plt.plot(x_fit, y_fit, linestyle=':', color='red', label='Best Fit Line')


    plt.plot(x, y)

    plt.title('Sample Line Graph')

    plt.xlabel('X axis')
    plt.ylabel('Y axis')

    plt.show()

def main(args):
    plot()
    # plot(
    #     outcome_file=args.outcome_file,
    #     outcome_type=args.outcome_type
    # )


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description=__doc__)
    # args = add_args(parser).parse_args()
    main("args")
