# Write distributions of physicochemical properties for training molecules.
setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = FALSE)
library(argparse)

# parse arguments
parser = ArgumentParser(prog = 'outer-write-outcome-distrs-break-SELFIES.R')
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
  rnn_type = 'LSTM',
  embedding_size = 128,
  hidden_size = 1024,
  n_layers = 3,
  dropout = 0,
  batch_size = 64,
  learning_rate = 0.001,
  
  constraints = c('c5', 'none'),
  subset = c('valid', 'invalid'),
  
  ## don't put in filename
  mol_sample_idx = 1,
  max_sampled_mols = 500000
)

# define input files: sample files
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
input_filenames = with(grid, paste0('sample-', mol_sample_idx,
                                    '-constraints=', constraints,
                                    '-', subset, '.csv'))
input_files = file.path(sample_dir, sample_dirnames, input_filenames)

# define output files
output_dir = file.path(base_dir, "outcome_distrs", "break_SELFIES")
if (!dir.exists(output_dir)) dir.create(output_dir)
output_dirnames = pmap_chr(grid, function(...) {
  current = tibble(...)
  current %>%
    dplyr::select(database:learning_rate) %>%
    map2(., names(.), ~ {
      paste0(.y, '=', .x)
    }) %>%
    paste0(collapse = '-')
})
output_filenames = with(grid, paste0('outcome-distrs-', 
                                     'constraints=', constraints,
                                     '-', subset, '.csv.gz'))
output_files = file.path(output_dir, output_dirnames, output_filenames)

# now, check for which parameters are already complete
grid0 = grid %>%
  mutate(input_file = input_files,
         output_file = output_files) %>% 
  filter(file.exists(input_file),
         !file.exists(output_file))
# remove columns that are all NA
keep = map_lgl(grid0, ~ any(!is.na(.x) & .x != 'NA'))
grid0 %<>% extract(, keep)

# write the grid that still needs to be run
grid_file = "sh/grids/write-outcome-distrs.txt"
grid_dir = dirname(grid_file)
if (!dir.exists(grid_dir))
  dir.create(grid_dir, recursive = TRUE)
write.table(grid0, grid_file, quote = FALSE, row.names = FALSE, sep = "\t")

# write the sh file dynamically
sh_file = 'sh/write-outcome-distrs/write-outcome-distrs-break-SELFIES.sh'
sh_dir = dirname(sh_file)
if (!dir.exists(sh_dir)) 
  dir.create(sh_dir, recursive = TRUE)
write_sh(job_name = 'write-outcome-distrs-break-SELFIES',
         sh_file = sh_file,
         grid_file = grid_file,
         inner_file = 'python/inner-write-outcome-distrs.py',
         system = system,
         time = 48,
         mem = 64)

# finally, run the job on whatever system we're on
submit_job(grid0, sh_file, args$allocation, system)
