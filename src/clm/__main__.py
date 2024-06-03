import logging
import argparse
import os
import clm
from clm.commands import (
    preprocess,
    create_training_sets,
    train_models_RNN,
    sample_molecules_RNN,
    tabulate_molecules,
    collect_tabulated_molecules,
    process_tabulated_molecules,
    write_structural_prior_CV,
    write_formula_prior_CV,
    calculate_outcomes,
    write_nn_Tc,
    prep_outcomes_freq,
    write_freq_distribution,
    train_discriminator,
    calculate_outcome_distrs,
    add_carbon,
    forecast,
    plot,
)


logger = logging.getLogger("clm")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=clm.__version__)

    modules = (
        preprocess,
        create_training_sets,
        train_models_RNN,
        sample_molecules_RNN,
        tabulate_molecules,
        collect_tabulated_molecules,
        process_tabulated_molecules,
        write_structural_prior_CV,
        write_formula_prior_CV,
        calculate_outcomes,
        write_nn_Tc,
        prep_outcomes_freq,
        write_freq_distribution,
        train_discriminator,
        calculate_outcome_distrs,
        add_carbon,
        forecast,
        plot,
    )

    subparsers = parser.add_subparsers(title="Choose a command")
    subparsers.required = True

    def get_str_name(module):
        return os.path.splitext(os.path.basename(module.__file__))[0]

    for module in modules:
        this_parser = subparsers.add_parser(
            get_str_name(module), description=module.__doc__
        )
        this_parser.add_argument(
            "-v", "--verbose", action="store_true", help="Increase verbosity"
        )
        module.add_args(this_parser)
        this_parser.set_defaults(func=module.main)

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info(f"CLM v{clm.__version__}")
    args.func(args)


if __name__ == "__main__":
    main()
