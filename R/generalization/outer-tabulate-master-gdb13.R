setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = F)
library(argparse)

# parse arguments
parser = ArgumentParser(prog = 'outer-tabulate-master-gdb13.R')
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
  learning_rate = 0.001,
  
  ## don't put in dirname
  n_samples = 10
) 

# define input directory
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

# define output files
output_files = with(grid, file.path(input_dirs, 'unique-masses-novel.tsv'))
time_files = with(grid, file.path(input_dirs, 'unique-masses-novel.time'))

# now, check for which parameters are already complete
grid0 = grid %>%
  mutate(input_dir = input_dirs,
         output_file = output_files,
         time_file = time_files) %>%
  mutate(keep = map2_lgl(input_dir, n_samples, ~ {
    files = file.path(.x, paste0('sample-', seq(1, .y), '-masses.tsv'))
    all(file.exists(files))
  })) %>% 
  filter(keep, !file.exists(time_file))

# write the grid that still needs to be run
grid_file = "sh/grids/tabulate-master.txt"
grid_dir = dirname(grid_file)
if (!dir.exists(grid_dir))
  dir.create(grid_dir, recursive = TRUE)
write.table(grid0, grid_file, quote = FALSE, row.names = FALSE, sep = "\t")

# write the sh file dynamically
sh_file = 'sh/gdb13/tabulate-master.sh'
sh_dir = dirname(sh_file)
if (!dir.exists(sh_dir)) 
  dir.create(sh_dir, recursive = TRUE)
write_sh(job_name = 'tabulate-master-gdb13',
         sh_file = sh_file,
         grid_file = grid_file,
         inner_file = 'R/prior/inner-tabulate-master.R',
         system = system,
         time = 8,
         mem = 120)

# finally, run the job on whatever system we're on
submit_job(grid0, sh_file, args$allocation, system)
