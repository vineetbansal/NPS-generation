# Tabulate valid molecules sampled from a trained chemical language model 
# by their canonical SMILES, exact mass, and molecular formula.
setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = F)
library(argparse)

# parse arguments
parser = ArgumentParser(prog = 'outer-tabulate-molecules.R')
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
  database = c('COCONUT', 'FooDB', 'LOTUS', 'NORMAN'),
  representation = c('SMILES', 'SELFIES'),
  enum_factor = 0,
  sample_idx = 1,
  k = 10,
  cv_fold = seq_len(10),
  
  # architecture - fixed
  rnn_type = 'LSTM',
  embedding_size = 128,
  hidden_size = 1024,
  n_layers = 3,
  dropout = 0,
  batch_size = 64,
  learning_rate = 0.001,
  
  ## don't put in dirname
  mol_sample_idx = seq_len(10),
  sample_mols = 10e6
)

# define training files (so we can filter each fold)
# define input files (*vocab)
train_dir = file.path(base_dir, "prior", "inputs")
train_filenames = pmap_chr(grid, function(...) {
  current = tibble(...)
  current %>%
    dplyr::select(database:cv_fold) %>%
    ## force canonical SMILES
    mutate(representation = 'SMILES', enum_factor = 0) %>% 
    map2(., names(.), ~ {
      paste0(.y, '=', .x)
    }) %>%
    paste0(collapse = '-') %>%
    paste0('train-', ., '.csv')
})
train_files = file.path(train_dir, train_filenames)

# define input files (sample files)
input_dir = file.path(base_dir, "prior", "sample")
input_dirnames = pmap_chr(grid, function(...) {
  current = tibble(...)
  current %>%
    dplyr::select(database:learning_rate) %>%
    map2(., names(.), ~ {
      paste0(.y, '=', .x)
    }) %>%
    paste0(collapse = '-')
})
input_files = with(grid, file.path(input_dir, input_dirnames,
                                   paste0('sample-', mol_sample_idx, '.csv')))
check_files = with(grid, file.path(input_dir, input_dirnames,
                                   paste0('time-', mol_sample_idx, '.csv')))

# create output files
output_files = with(grid, file.path(input_dir, input_dirnames,
                                    paste0('sample-', mol_sample_idx, 
                                           '-masses.tsv')))
# time files
time_files = gsub("\\.tsv$", ".time", output_files)

# _now_ cross with sample_idx and check for which parameters are already complete
grid0 = grid %>% 
  mutate(train_file = train_files,
         input_file = input_files,
         check_file = check_files,
         output_file = output_files,
         time_file = time_files) %>% 
  filter(file.exists(input_file),
         file.exists(train_file),
         file.exists(check_files),
         !file.exists(time_file))

# write the grid that still needs to be run
grid_file = "sh/grids/tabulate-molecules.txt"
grid_dir = dirname(grid_file)
if (!dir.exists(grid_dir))
  dir.create(grid_dir, recursive = TRUE)
write.table(grid0, grid_file, quote = FALSE, row.names = FALSE, sep = "\t")

# write the sh file dynamically
sh_file = 'sh/tabulate-molecules/tabulate-molecules-prior.sh'
sh_dir = dirname(sh_file)
if (!dir.exists(sh_dir)) 
  dir.create(sh_dir, recursive = TRUE)
write_sh(job_name = 'tabulate-molecules-prior',
         sh_file = sh_file,
         grid_file = grid_file,
         inner_file = 'python/inner-tabulate-molecules.py',
         system = system,
         time = 120,
         mem = 16)

# finally, run the job on whatever system we're on
submit_job(grid0, sh_file, args$allocation, system)
