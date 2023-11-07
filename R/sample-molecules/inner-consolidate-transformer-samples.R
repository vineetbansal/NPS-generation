setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = FALSE)
library(argparse)

# dynamically parse arguments
parser = ArgumentParser(prog = 'inner-consolidate-transformer-samples.R')
grid = read.delim("sh/grids/consolidate-transformer-samples.txt")
for (param_name in colnames(grid))
  parser$add_argument(paste0('--', param_name),
                      type = typeof(grid[[param_name]]))
args = parser$parse_args()
print(args)

library(tidyverse)
library(magrittr)

# parse files
input_files = map_chr(seq_len(10), ~ gsub('\\$', .x, args$input_file))

# double-check lengths
sample_sizes = map_int(input_files, ~ paste('wc -l', .x) %>% 
                         system(intern = TRUE) %>% 
                         gsub(" .*$", "", .) %>% 
                         as.integer())
if (!all(sample_sizes >= 1e6))
  stop('sampling not done!')

# build command
cmd = paste('cat', paste(input_files, collapse = ' '), '>', args$output_file)
system(cmd)

# check file size
final_size = paste('wc -l', args$output_file) %>% 
  system(intern = TRUE) %>% 
  gsub(" .*$", "", .) %>% 
  as.integer()
if (final_size < 10e6)
  stop('WARNING: final sample is <10e6! ', final_size)
