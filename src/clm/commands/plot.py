import argparse
from clm.plot.calculate_outcomes import plot as calculate_outcomes


def add_args(parser):
    parser.add_argument(
        "evaluation_type",
        type=str,
        help="Type of evaluation you want figures of. Valid options are:  \n"
        " calculate_outcomes",
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
