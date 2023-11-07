# Calculate outcomes for transforrmer-based models of SELFIES 
# strings that have been decoded without valence constraints. 
setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = FALSE)
library(argparse)

# parse arguments
parser = ArgumentParser(prog = 'outer-calculate-outcomes-transformer-SELFIES-valid.R')
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
  representation = 'SELFIES', 
  enum_factor = 0,
  n_molecules = 1e5,
  min_tc = 0,
  sample_idx = seq_len(10),
  
  # architecture - fixed
  embedding_size = 256,
  n_blocks = 8,
  n_heads = 8,
  max_len = 250,
  dropout = 0,
  exp_factor = 4,
  batch_size = 64,
  learning_rate = 0.0005,
  
  # SELFIES constraints
  constraints = c('none', 'c5'),
  
  ## don't put in filename
  max_orig_mols = 500000,
  max_sampled_mols = 500000,
  mol_sample_idx = 0
)

# define training files
train_dir = file.path(base_dir, "training_sets")
train_filenames = pmap_chr(grid, function(...) {
  current = tibble(...)
  current %>%
    ## read training set as SMILES! ##
    mutate(representation = 'SMILES') %>% 
    dplyr::select(database:sample_idx) %>%
    map2(., names(.), ~ {
      paste0(.y, '=', .x)
    }) %>%
    paste0(collapse = '-') %>%
    paste0('input-', ., '.smi')
})
train_files = file.path(train_dir, train_filenames)

# define input files (samples, filtered by likelihood)
sample_dir = file.path(base_dir, "sample")
sample_dirnames = pmap_chr(grid, function(...) {
  current = tibble(...)
  current %>%
    dplyr::select(database:learning_rate) %>%
    map2(., names(.), ~ {
      paste0(.y, '=', .x)
    }) %>%
    paste0(collapse = '-')
})
valid_filenames = with(grid, paste0('sample-', mol_sample_idx, 
                                    '-constraints=', constraints,
                                    '-valid.csv'))
time_filenames = with(grid, paste0('sample-', mol_sample_idx, 
                                   '-constraints=', constraints,
                                   '-time.csv'))
valid_files = file.path(sample_dir, sample_dirnames, valid_filenames)
time_files = file.path(sample_dir, sample_dirnames, time_filenames)

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
output_files = grid %$%
  file.path(output_dir, output_dirnames, 
            paste0('outcomes-constraints=', constraints, ".csv"))

# now, check for which parameters are already complete
grid0 = grid %>%
  mutate(train_file = train_files,
         sampled_file = valid_files,
         time_file = time_files,
         output_file = output_files) %>% 
  filter(file.exists(train_file),
         file.exists(sampled_file),
         file.exists(time_file),
         !file.exists(output_file)) %>% 
  # reset representation for input to python clean_mols
  mutate(representation = 'SMILES')
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
sh_file = 'sh/calculate-outcomes/calculate-outcomes-transformer-SELFIES-valid.sh'
sh_dir = dirname(sh_file)
if (!dir.exists(sh_dir)) 
  dir.create(sh_dir, recursive = TRUE)
write_sh(job_name = 'calculate-outcomes-transformer-SELFIES-valid',
         sh_file = sh_file,
         grid_file = grid_file,
         inner_file = 'python/inner-calculate-outcomes.py',
         system = system,
         time = 12,
         mem = 200)

# finally, run the job on whatever system we're on
submit_job(grid0, sh_file, args$allocation, system)
