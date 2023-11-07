setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = FALSE)
library(argparse)

# parse arguments
parser = ArgumentParser(prog = 'outer-write-formula-prior-break-SELFIES.R')
parser$add_argument('--allocation', type = 'character', default = "root")
args = parser$parse_args()

library(tidyverse)
library(magrittr)

# load grid functions
source("R/functions/submit_job.R")
source("R/functions/write_sh.R")
source("R/functions/detect_system.R")

# define PubChem file
pubchem_file = file.path(base_dir, 'PubChem/PubChem.tsv')

# set up grid
grid = tidyr::crossing(
  database = c('COCONUT', 'FooDB', 'LOTUS', 'NORMAN'),
  representation = 'SELFIES',
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
  
  constraints = 'none'
)

# define model sample files
sample_dir = file.path(base_dir, "prior", "sample")
sample_dirnames = pmap_chr(grid, function(...) {
  current = tibble(...)
  current %>%
    dplyr::select(database:learning_rate) %>%
    map2(., names(.), ~ {
      paste0(.y, '=', .x)
    }) %>%
    paste0(collapse = '-')
})
sample_files = grid %$% 
  file.path(sample_dir, sample_dirnames,
            paste0('unique-masses-novel-constraints=', constraints, '.tsv'))

# define training files (so we can filter each fold)
train_dir = file.path(base_dir, "prior", "inputs")
train_filenames = pmap_chr(grid, function(...) {
  current = tibble(...)
  current %>%
    ## force canonical SMILES
    mutate(representation = 'SMILES',
           enum_factor = 0) %>% 
    dplyr::select(database:cv_fold) %>%
    map2(., names(.), ~ {
      paste0(.y, '=', .x)
    }) %>%
    paste0(collapse = '-') %>%
    paste0('train-', ., '.csv')
})
train_files = file.path(train_dir, train_filenames)
test_files = gsub("train-", "test-", train_files)

# define output files
output_dir = file.path(base_dir, "prior", "formula_prior", "break_SELFIES")
output_dirnames = basename(dirname(sample_files))
output_dirs = file.path(output_dir, output_dirnames)
output_files = file.path(output_dirs, 'CV-ranks.csv.gz')

# now, check for which parameters are already complete
grid0 = grid %>%
  mutate(sample_file = sample_files,
         train_file = train_files,
         test_file = test_files,
         pubchem_file = pubchem_file, 
         output_dir = output_dirs,
         output_file = output_files) %>% 
  # set ppm error
  mutate(err_ppm = 10) %>% 
  filter(file.exists(sample_file),
         file.exists(train_file),
         file.exists(test_file),
         !file.exists(output_file))

# write the grid that still needs to be run
grid_file = "sh/grids/write-formula-prior.txt"
grid_dir = dirname(grid_file)
if (!dir.exists(grid_dir))
  dir.create(grid_dir, recursive = TRUE)
write.table(grid0, grid_file, quote = FALSE, row.names = FALSE, sep = "\t")

# write the sh file dynamically
sh_file = 'sh/prior/write-formula-prior-break-SELFIES.sh'
sh_dir = dirname(sh_file)
if (!dir.exists(sh_dir)) 
  dir.create(sh_dir, recursive = TRUE)
write_sh(job_name = 'formula-CV',
         sh_file = sh_file,
         grid_file = grid_file,
         inner_file = 'python/inner-write-formula-prior-CV.py',
         system = system,
         time = 72,
         mem = 96)

# finally, run the job on whatever system we're on
submit_job(grid0, sh_file, args$allocation, system)
