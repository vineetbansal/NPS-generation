# Calculate Cohen's d (continuous) / odds ratio (discrete) for physicochemical
# property distributions of sampled molecules, compared to the training set.
setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = FALSE)
library(argparse)

# parse arguments
parser = ArgumentParser(prog = 'outer-calculate-effect-sizes.R')
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
  database = 'ChEMBL',
  representation = c('SMILES', 'SELFIES'),
  enum_factor = 0,
  n_molecules = 1e5,
  min_tc = 0,
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

# define outcome files for sampled molecules
sample_dir = file.path(base_dir, "outcome_distrs")
if (!dir.exists(sample_dir)) dir.create(sample_dir)
sample_dirnames = pmap_chr(grid, function(...) {
  current = tibble(...)
  current %>%
    dplyr::select(database:learning_rate) %>%
    map2(., names(.), ~ {
      paste0(.y, '=', .x)
    }) %>%
    paste0(collapse = '-')
})
sample_files = file.path(sample_dir, sample_dirnames, 'outcome-distrs.csv.gz')

# define outcome files for training molecules
train_dir = file.path(base_dir, "outcome_distrs", "training")
if (!dir.exists(train_dir)) dir.create(train_dir)
train_dirnames = pmap_chr(grid, function(...) {
  current = tibble(...)
  current %>%
    ## force SMILES
    mutate(representation = 'SMILES') %>% 
    dplyr::select(database:sample_idx) %>%
    map2(., names(.), ~ {
      paste0(.y, '=', .x)
    }) %>%
    paste0(collapse = '-')
})
train_files = file.path(train_dir, train_dirnames, 'outcome-distrs.csv.gz')

# last, define output files
output_files = file.path(sample_dir, sample_dirnames, 'effect-sizes.rds')

# now, check for which parameters are already complete
grid0 = grid %>%
  mutate(sample_file = sample_files,
         train_file = train_files,
         output_file = output_files) %>% 
  filter(file.exists(sample_file),
         file.exists(train_file),
         !file.exists(output_file))
# remove columns that are all NA
keep = map_lgl(grid0, ~ any(!is.na(.x) & .x != 'NA'))
grid0 %<>% extract(, keep)

# write the grid that still needs to be run
grid_file = "sh/grids/calculate-effect-sizes.txt"
grid_dir = dirname(grid_file)
if (!dir.exists(grid_dir))
  dir.create(grid_dir, recursive = TRUE)
write.table(grid0, grid_file, quote = FALSE, row.names = FALSE, sep = "\t")

# write the sh file dynamically
sh_file = 'sh/write-outcome-distrs/calculate-effect-sizes.sh'
sh_dir = dirname(sh_file)
if (!dir.exists(sh_dir)) 
  dir.create(sh_dir, recursive = TRUE)
write_sh(job_name = 'calculate-effect-sizes',
         sh_file = sh_file,
         grid_file = grid_file,
         inner_file = 'R/outcome-distrs/inner-calculate-effect-sizes.R',
         system = system,
         time = 24,
         mem = 48)

# finally, run the job on whatever system we're on
submit_job(grid0, sh_file, args$allocation, system)
