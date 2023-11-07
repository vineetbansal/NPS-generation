setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = FALSE)
library(argparse)

# dynamically parse arguments
parser = ArgumentParser(prog = 'inner-evaluate-generalization.R')
grid = read.delim("sh/grids/evaluate-generalization.txt")
for (param_name in colnames(grid))
  parser$add_argument(paste0('--', param_name),
                      type = typeof(grid[[param_name]]))
args = parser$parse_args()
print(args)

library(tidyverse)
library(magrittr)
library(data.table)
source("R/functions/detect_system.R")

# read the entire GDB13 sample
gdb_file = file.path(base_dir, 'GDB13', 'gdb13_canonical.smi')
gdb = fread(gdb_file, header = FALSE, col.names = 'smiles')$smiles

# remove training set molecules
train_file = file.path(base_dir, 'GDB13', 'inputs', 
                       'gdb13-1M-representation=SMILES-enum_factor=0.csv')
gdb_sample = readLines(train_file)
gdb %<>% setdiff(gdb_sample)

# read the model output
sample = read_csv(args$input_file)

# some summary statistics:
# - number of unique molecules
n_unique = nrow(sample)
# - number of valid (total) molecules
n_total = sum(sample$size)
# - number of molecules in GDB13
n_in_gdb = sum(sample$smiles %in% gdb)
# - number of molecules not in GDB13
n_not_in_gdb = sum(!sample$smiles %in% gdb)
# combine
summary_df = data.frame(outcome = c('unique molecules',
                                    'total molecules',
                                    'molecules in GDB',
                                    'molecules not in GDB',
                                    'GDB size'),
                        value = c(n_unique, n_total, n_in_gdb, n_not_in_gdb,
                                  length(gdb)))

# main outcome: % of GDB that we recapitulate 
increments = seq(0, nrow(sample), 1e6)
pct_of_gdb = map_dbl(increments, ~ {
  increment = .x
  sample_idxs = sample.int(nrow(sample), increment, replace = TRUE, 
                          prob = sample$size)
  sample0 = sample[sample_idxs, ]
  mean(gdb %in% sample0$smiles)
})
curve_df = data.frame(n_molecules = increments,
                      pct_of_gdb = pct_of_gdb)

# combine
outputs = list(summary = summary_df, curve = curve_df)
output_dir = dirname(args$output_file)
if (!dir.exists(output_dir))
  dir.create(output_dir, recursive = TRUE)
saveRDS(outputs, args$output_file)
