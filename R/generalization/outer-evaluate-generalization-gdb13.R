setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = F)
library(argparse)

# parse arguments
parser = ArgumentParser(prog = 'outer-evaluate-generalization-gdb13.R')
parser$add_argument('--allocation', type = 'character', default = "root")
args = parser$parse_args()

library(tidyverse)
library(magrittr)

# load grid functions
source("R/functions/submit_job.R")
source("R/functions/write_sh.R")
source("R/functions/detect_system.R")

# set up grid
grid = tidyr::crossing(
  representation = c('SMILES', 'SELFIES'),
  enum_factor = 0,
  sample_idx = seq_len(10),
  
  # architecture - fixed
  rnn_type = 'LSTM',
  embedding_size = 128,
  hidden_size = 1024,
  n_layers = 3,
  dropout = 0,
  batch_size = 64,
  learning_rate = 0.001
) 

# define input files
input_dir = file.path(base_dir, "GDB13", "sample")
input_dirnames = pmap_chr(grid, function(...) {
  current = tibble(...)
  current %>%
    dplyr::select(representation:learning_rate) %>%
    map2(., names(.), ~ {
      paste0(.y, '=', .x)
    }) %>%
    paste0(collapse = '-')
})
input_dirs = file.path(input_dir, input_dirnames)
input_files = file.path(input_dirs, 'unique-masses-novel.tsv')
check_files = file.path(input_dirs, 'unique-masses-novel.time')

# create output files
output_dir = file.path(base_dir, "GDB13", "outcomes")
if (!dir.exists(output_dir)) dir.create(output_dir)
output_files = file.path(output_dir, input_dirnames, 'outcomes.rds')

# now, check for which parameters are already complete
grid0 = grid %>%
  mutate(input_file = input_files,
         check_file = check_files,
         output_file = output_files) %>%
  filter(file.exists(input_file),
         file.exists(check_files),
         !file.exists(output_files))

# write the grid that still needs to be run
grid_file = "sh/grids/evaluate-generalization.txt"
grid_dir = dirname(grid_file)
if (!dir.exists(grid_dir))
  dir.create(grid_dir, recursive = TRUE)
write.table(grid0, grid_file, quote = FALSE, row.names = FALSE, sep = "\t")

# write the sh file dynamically
sh_file = 'sh/gdb13/evaluate-generalization.sh'
sh_dir = dirname(sh_file)
if (!dir.exists(sh_dir)) 
  dir.create(sh_dir, recursive = TRUE)
write_sh(job_name = 'evaluate-generalization-gdb13',
         sh_file = sh_file,
         grid_file = grid_file,
         inner_file = 'R/generalization/inner-evaluate-generalization.R',
         system = system,
         time = 24,
         mem = 180)

# finally, run the job on whatever system we're on
submit_job(grid0, sh_file, args$allocation, system)
