setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = FALSE)
library(argparse)

# parse arguments
parser = ArgumentParser(prog = 'outer-calculate-effect-sizes-RNN.R')
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
  representation = 'SMILES',
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
  mol_sample_idx = 1,
  max_mols = 10e6
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

# define sample files
sample_dir = file.path(base_dir, "sample")
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
check_files = with(grid, file.path(sample_dir, sample_dirnames,
                                   paste0('time-', mol_sample_idx, '.csv')))

# define validity files
valid_files = with(grid, file.path(sample_dir, sample_dirnames,
                                   paste0('sample-', mol_sample_idx, 
                                          '-valid.csv')))
time_files = with(grid, file.path(sample_dir, sample_dirnames,
                                  paste0('sample-', mol_sample_idx, 
                                         '-valid-time.csv')))

# now, define output files: 1. cohen's d, 2. error frequencies
output_dir = file.path(base_dir, 'parse-valid-smiles')
effect_filenames = paste0(basename(dirname(valid_files)), '__', 
                          basename(valid_files) %>% 
                            gsub("\\.csv", "-effect.csv", .))
effect_files = file.path(output_dir, effect_filenames)
freq_filenames = paste0(basename(dirname(valid_files)), '__', 
                        basename(valid_files) %>% 
                          gsub("\\.csv", "-freqs.csv", .))
freq_files = file.path(output_dir, freq_filenames)

# now, check for which parameters are already complete
grid0 = grid %>%
  mutate(valid_file = valid_files,
         time_file = time_files,
         effect_file = effect_files,
         freq_file = freq_files) %>% 
  filter(file.exists(valid_file),
         file.exists(time_file),
         !file.exists(effect_file) |
           !file.exists(freq_file))
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
sh_file = 'sh/sample-molecules/calculate-effect-sizes-RNN.sh'
sh_dir = dirname(sh_file)
if (!dir.exists(sh_dir)) 
  dir.create(sh_dir, recursive = TRUE)
write_sh(job_name = 'calculate-effect-sizes-RNN',
         sh_file = sh_file,
         grid_file = grid_file,
         inner_file = 'R/sample-molecules/inner-calculate-effect-sizes.R',
         system = system,
         time = 3,
         mem = 32)

# finally, run the job on whatever system we're on
submit_job(grid0, sh_file, args$allocation, system)
