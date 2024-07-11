import pandas as pd
from clm.functions import read_csv_file, write_to_csv_file


def add_args(parser):
    parser.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        help="Input file paths to collapse",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="File path to save the collapsed file",
    )
    parser.add_argument(
        "--has_header",
        action="store_true",
    )
    return parser


def collapse_files(input_files, output_file, has_header=False):
    dataframes = []
    for input_file in input_files:
        dataframe = read_csv_file(
            input_file, header=0 if has_header else None, dtype=str
        )
        dataframes.append(dataframe)
    data = pd.concat(dataframes)

    data = data.drop_duplicates()
    data = data.sort_values(by=data.columns.tolist())

    write_to_csv_file(output_file, data)


def main(args):
    collapse_files(
        input_files=args.carbon_files,
        output_file=args.output_file,
        has_header=args.has_header,
    )
