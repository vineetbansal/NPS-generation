# Calculate outcomes for recurrent neural network-based models of SMILES or
# SELFIES strings.
setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = FALSE)
library(argparse)

# parse arguments
parser = ArgumentParser(prog = 'outer-calculate-outcomes-RNN.R')
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
  database = c('GDB13', 'ChEMBL'),
  representation = c('SMILES', 'SELFIES'),
  enum_factor = c(0, 10, 30),
  n_molecules = c(3e4, 1e5, 3e5),
  min_tc = seq(0, 0.15, 0.05),
  sample_idx = seq_len(10),
    
  # architecture - fixed
  rnn_type = 'LSTM',
  embedding_size = 128,
  hidden_size = 1024,
  n_layers = 3,
  dropout = 0,
  batch_size = 64,
  learning_rate = 0.001,
  
  ## don't put in filename
  max_orig_mols = 500000,
  max_sampled_mols = 500000
)

# take only one step away from the default at a time
defaults = list(database = 'ChEMBL',
                representation = c('SMILES', 'SELFIES'),
                enum_factor = 0,
                n_molecules = 1e5,
                min_tc = 0)
n_steps = grid %>% 
  pmap_int(function(...) {
    current = tibble(...)
    names = names(defaults)
    map2_int(current[, names], defaults, ~ .x %in% .y) %>% 
      sum()
  })
keep = which(n_steps >= (length(defaults) - 1))
grid %<>% extract(keep, )

# define training files
train_dir = file.path(base_dir, "training_sets")
train_filenames = pmap_chr(grid, function(...) {
  current = tibble(...)
  current %>%
    dplyr::select(database:sample_idx) %>%
    map2(., names(.), ~ {
      paste0(.y, '=', .x)
    }) %>%
    paste0(collapse = '-') %>%
    paste0('input-', ., '.smi')
})
train_files = file.path(train_dir, train_filenames)

# define model files: smiles file, timing file
model_dir = file.path(base_dir, 'models')
model_dirnames = pmap_chr(grid, function(...) {
  current = tibble(...)
  current %>%
    dplyr::select(database:learning_rate) %>%
    map2(., names(.), ~ {
      paste0(.y, '=', .x)
    }) %>%
    paste0(collapse = '-')
})
smiles_files = file.path(model_dir, model_dirnames, 'SMILES.smi')
time_files = file.path(model_dir, model_dirnames, 'timing.csv')

# define output files
output_dir = file.path(base_dir, "outcomes")
output_dirnames = pmap_chr(grid, function(...) {
  current = tibble(...)
  current %>%
    dplyr::select(database:learning_rate) %>%
    map2(., names(.), ~ {
      paste0(.y, '=', .x)
    }) %>%
    paste0(collapse = '-')
})
output_files = file.path(output_dir, output_dirnames, 'outcomes.csv')

# now, check for which parameters are already complete
grid0 = grid %>%
  mutate(train_file = train_files,
         sampled_file = smiles_files,
         time_file = time_files,
         output_file = output_files) %>% 
  filter(file.exists(train_file),
         file.exists(sampled_file),
         file.exists(time_file),
         !file.exists(output_file))
# remove columns that are all NA
keep = map_lgl(grid0, ~ any(!is.na(.x) & .x != 'NA'))
grid0 %<>% extract(, keep)

# write the grid that still needs to be run
grid_file = "sh/grids/calculate-outcomes.txt"
grid_dir = dirname(grid_file)
if (!dir.exists(grid_dir))
  dir.create(grid_dir, recursive = TRUE)
write.table(grid0, grid_file, quote = FALSE, row.names = FALSE, sep = "\t")

# write the sh file dynamically
sh_file = 'sh/calculate-outcomes/calculate-outcomes-RNN.sh'
sh_dir = dirname(sh_file)
if (!dir.exists(sh_dir)) 
  dir.create(sh_dir, recursive = TRUE)
write_sh(job_name = 'calculate-outcomes-RNN',
         sh_file = sh_file,
         grid_file = grid_file,
         inner_file = 'python/inner-calculate-outcomes.py',
         system = system,
         time = 8,
         mem = 400)

# finally, run the job on whatever system we're on
submit_job(grid0, sh_file, args$allocation, system)
