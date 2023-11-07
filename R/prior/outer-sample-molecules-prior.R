# Sample molecules from chemical language models trained on the prior training
# datasets.
setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = FALSE)
library(argparse)

# parse arguments
parser = ArgumentParser(prog = 'outer-sample-molecules-prior.R')
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

# define input files (*vocab)
input_dir = file.path(base_dir, "prior", "inputs")
input_filenames = pmap_chr(grid, function(...) {
  current = tibble(...)
  current %>%
    dplyr::select(database:cv_fold) %>%
    map2(., names(.), ~ {
      paste0(.y, '=', .x)
    }) %>%
    paste0(collapse = '-') %>%
    paste0('train-', ., '.csv')
})
input_files = file.path(input_dir, input_filenames)
vocab_files = gsub("\\.csv$", ".vocabulary", input_files)

# define model output files: *model file, loss file, smiles file, timing file
output_dir = file.path(base_dir, "prior", "models")
output_dirnames = pmap_chr(grid, function(...) {
  current = tibble(...)
  current %>%
    dplyr::select(database:learning_rate) %>%
    map2(., names(.), ~ {
      paste0(.y, '=', .x)
    }) %>%
    paste0(collapse = '-')
})
model_files = file.path(output_dir, output_dirnames, 'model.pt')
check_files = file.path(output_dir, output_dirnames, 'timing.csv')

# define sample output files
sample_dir = file.path(base_dir, "prior", "sample")
if (!dir.exists(sample_dir))
  dir.create(sample_dir)
sample_dirnames = pmap_chr(grid, function(...) {
  current = tibble(...)
  current %>%
    dplyr::select(database:learning_rate) %>%
    map2(., names(.), ~ {
      paste0(.y, '=', .x)
    }) %>%
    paste0(collapse = '-')
})
sample_files = with(grid, file.path(sample_dir, sample_dirnames,
                                    paste0('sample-', mol_sample_idx, '.csv')))
time_files = with(grid, file.path(sample_dir, sample_dirnames,
                                  paste0('time-', mol_sample_idx, '.csv')))
nvstat_files = with(grid, file.path(sample_dir,
                                    paste0(sample_dirnames, '.nvstat')))

# now, check for which parameters are already complete
grid0 = grid %>%
  mutate(input_file = input_files,
         vocab_file = vocab_files,
         model_file = model_files,
         check_file = check_files,
         output_file = sample_files,
         time_file = time_files,
         nvstat_file = nvstat_files) %>% 
  filter(file.exists(input_file),
         file.exists(vocab_file),
         file.exists(model_file),
         file.exists(check_file),
         ## time file is the last to be written
         !file.exists(time_file))
# remove columns that are all NA
keep = map_lgl(grid0, ~ any(!is.na(.x) & .x != 'NA'))
grid0 %<>% extract(, keep)

# write the grid that still needs to be run
grid_file = "sh/grids/sample-molecules.txt"
grid_dir = dirname(grid_file)
if (!dir.exists(grid_dir))
  dir.create(grid_dir, recursive = TRUE)
write.table(grid0, grid_file, quote = FALSE, row.names = FALSE, sep = "\t")

# write the sh file dynamically
sh_file = 'sh/sample-molecules/sample-molecules-prior.sh'
sh_dir = dirname(sh_file)
if (!dir.exists(sh_dir)) 
  dir.create(sh_dir, recursive = TRUE)
write_sh(job_name = 'sample-molecules-RNN',
         sh_file = sh_file,
         grid_file = grid_file,
         inner_file = 'python/inner-sample-molecules.py',
         system = system,
         time = 12,
         mem = 12,
         gpu = TRUE)

# finally, run the job on whatever system we're on
if (!grepl("-gpu$", args$allocation))
  args$allocation %<>% paste0('-gpu')
submit_job(grid0, sh_file, args$allocation, system, jobs_per_array = 10)
