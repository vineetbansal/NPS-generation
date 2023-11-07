setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = FALSE)
library(argparse)

# dynamically parse arguments
parser = ArgumentParser(prog = 'inner-tabulate-master.R')
grid = read.delim("sh/grids/tabulate-master.txt")
for (param_name in colnames(grid))
  parser$add_argument(paste0('--', param_name),
                      type = typeof(grid[[param_name]]))
args = parser$parse_args()
print(args)

library(tidyverse)
library(magrittr)
library(data.table)

# create output directory if it doesn't exist
output_dir = dirname(args$output_file)
if (!dir.exists(output_dir))
  dir.create(output_dir, recursive = TRUE)

# start timer
start_time = Sys.time()

# list input files
input_filenames = paste0('sample-', seq_len(args$n_samples), '-masses.tsv')
input_files = file.path(args$input_dir, input_filenames)

# read them all
dats = map(input_files, fread)
dat = bind_rows(dats)

# group by canonical SMILES
smiles = setDT(dat)[, .(size = sum(size)), .(smiles, mass, formula)]

# write
fwrite(smiles, file = args$output_file)

# write time
end_time = Sys.time()
total_time = print(end_time - start_time)
writeLines(as.character(total_time), args$time_file)
