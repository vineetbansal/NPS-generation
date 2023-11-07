setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = FALSE)
library(argparse)

# parse arguments
parser = ArgumentParser(prog = 'outer-train-models-gdb13.R')
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
  
  ## don't put in filename
  max_epochs = 99999,
  patience = 50000,
  log_every_steps = 100,
  log_every_epochs = 1,
  sample_mols = 500000
)

# define input files
input_dir = file.path(base_dir, "GDB13", "inputs")
input_filenames = pmap_chr(grid, function(...) {
  current = tibble(...)
  current %>%
    dplyr::select(representation:enum_factor) %>%
    map2(., names(.), ~ {
      paste0(.y, '=', .x)
    }) %>%
    paste0(collapse = '-') %>%
    paste0('gdb13-1M-', ., '.csv')
})
input_files = file.path(input_dir, input_filenames)
vocab_files = gsub("\\.csv$", ".vocabulary", input_files)

# define output files: model file, loss file, smiles file, timing file
output_dir = file.path(base_dir, "GDB13", "models")
if (!dir.exists(output_dir))
  dir.create(output_dir)
output_dirnames = pmap_chr(grid, function(...) {
  current = tibble(...)
  current %>%
    dplyr::select(representation:learning_rate) %>%
    map2(., names(.), ~ {
      paste0(.y, '=', .x)
    }) %>%
    paste0(collapse = '-')
})
smiles_files = file.path(output_dir, output_dirnames, 'SMILES.smi')
model_files = file.path(output_dir, output_dirnames, 'model.pt')
loss_files = file.path(output_dir, output_dirnames, 'loss.csv')
time_files = file.path(output_dir, output_dirnames, 'timing.csv')

# now, check for which parameters are already complete
grid0 = grid %>%
  mutate(input_file = input_files,
         vocab_file = vocab_files,
         smiles_file = smiles_files,
         model_file = model_files,
         loss_file = loss_files,
         time_file = time_files) %>% 
  filter(file.exists(input_file),
         file.exists(vocab_file),
         ## time file is the last to be written
         !file.exists(time_file))
# remove columns that are all NA
keep = map_lgl(grid0, ~ any(!is.na(.x) & .x != 'NA'))
grid0 %<>% extract(, keep)

# write the grid that still needs to be run
grid_file = "sh/grids/train-models.txt"
grid_dir = dirname(grid_file)
if (!dir.exists(grid_dir))
  dir.create(grid_dir, recursive = TRUE)
write.table(grid0, grid_file, quote = FALSE, row.names = FALSE, sep = "\t")

# write the sh file dynamically
sh_file = 'sh/train-models/train-models-gdb13.sh'
sh_dir = dirname(sh_file)
if (!dir.exists(sh_dir)) 
  dir.create(sh_dir, recursive = TRUE)
write_sh(job_name = 'train-models-gdb13',
         sh_file = sh_file,
         grid_file = grid_file,
         inner_file = 'python/inner-train-models.py',
         system = system,
         time = 168,
         mem = 16,
         gpu = TRUE)

# finally, run the job on whatever system we're on
if (!grepl("-gpu$", args$allocation))
  args$allocation %<>% paste0('-gpu')
submit_job(grid0, sh_file, args$allocation, system, jobs_per_array = 10)
