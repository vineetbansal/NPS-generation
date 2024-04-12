import argparse
from clm.plot.calculate_outcomes import plot as calculate_outcomes
from clm.plot.write_nn_tc import plot as write_nn_tc
from clm.plot.train_discriminator import plot as train_discriminator
from clm.plot.freq_distribution import plot as freq_distribution
from clm.plot.calculate_outcome_distrs import plot as calculate_outcome_distrs

def add_args(parser):
    parser.add_argument(
        "evaluation_type",
        type=str,
        help="Type of evaluation you want figures of. Valid options are:  \n"
        " calculate_outcomes, write_nn_tc, train_discriminator, freq_distribution \n",
    )
    parser.add_argument(
        "--outcome_dir",
        type=str,
        required=True,
        help="Path to directory where all the model evaluation files are saved ",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to directory to save resulting images(s) at",
    )
    return parser


def plot(evaluation_type, outcome_dir, output_dir):
    if evaluation_type == "calculate_outcomes":
        calculate_outcomes(outcome_dir, output_dir)
    elif evaluation_type == "write_nn_tc":
        write_nn_tc(outcome_dir, output_dir)
    elif evaluation_type == "train_discriminator":
        train_discriminator(outcome_dir, output_dir)
    elif evaluation_type == "freq_distribution":
        freq_distribution(outcome_dir, output_dir)
    elif evaluation_type == "calculate_outcome_distrs":
        calculate_outcome_distrs(outcome_dir, output_dir)


def main(args):
    plot(
        evaluation_type=args.evaluation_type,
        outcome_dir=args.outcome_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    main(args)
