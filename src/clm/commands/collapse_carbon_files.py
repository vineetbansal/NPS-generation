from clm.functions import read_csv_file, write_to_csv_file


def add_args(parser):
    parser.add_argument(
        "--carbon_files",
        type=str,
        nargs="+",
        help="Input file paths for compressed carbon files",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="File path to save the all the compressed carbon files",
    )
    return parser


def collapse_carbon_files(carbon_files, output_file):
    carbon_files = sorted(carbon_files)
    header_list = read_csv_file(carbon_files[0]).columns.tolist()
    header = "".join(header_list)
    print(header)
    write_to_csv_file(output_file, str(header) + "\n", "a+")
    lines = set()
    for file in carbon_files:
        for i, line in enumerate(read_csv_file(file, delimiter="\t").values.tolist()):
            lines.add(tuple(line))
    sorted_lines = sorted(lines)
    for line in sorted_lines:
        input_smiles, mut_can, mass, formula, mut_inchi = line
        row = "\t".join([input_smiles, mut_can, str(mass), formula, mut_inchi]) + "\n"
        write_to_csv_file(output_file, row, "a+")


def main(args):
    collapse_carbon_files(carbon_files=args.carbon_files, output_file=args.output_file)
