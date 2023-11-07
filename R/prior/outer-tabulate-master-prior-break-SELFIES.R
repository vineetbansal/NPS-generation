setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = F)
library(argparse)

# parse arguments
parser = ArgumentParser(prog = 'outer-tabulate-master-prior-break-SELFIES.R')
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
  
  constraints = 'none',
  
  ## don't put in dirname
  n_samples = 10
) 

# define input directory
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
input_dirs = file.path(input_dir, input_dirnames)

# define output files
output_files = grid %$%
  file.path(input_dirs, paste0("unique-masses-novel-constraints=", constraints,
                               ".tsv"))
time_files = gsub("\\.tsv$", ".time", output_files)

# now, check for which parameters are already complete
grid0 = grid %>%
  mutate(input_dir = input_dirs,
         output_file = output_files,
         time_file = time_files)
keep = pmap_lgl(grid0, function(...) {
  current = tibble(...)
  files = file.path(current$input_dir,
                    paste0('sample-', seq(1, current$n_samples), 
                           '-constraints=', current$constraints, '-masses.tsv'))
  all(file.exists(files))
})
grid0 %<>% mutate(keep = keep) %>% 
  filter(keep, !file.exists(time_file))

# write the grid that still needs to be run
grid_file = "sh/grids/tabulate-master-break-SELFIES.txt"
grid_dir = dirname(grid_file)
if (!dir.exists(grid_dir))
  dir.create(grid_dir, recursive = TRUE)
write.table(grid0, grid_file, quote = FALSE, row.names = FALSE, sep = "\t")

# write the sh file dynamically
sh_file = 'sh/prior/tabulate-master-break-SELFIES.sh'
sh_dir = dirname(sh_file)
if (!dir.exists(sh_dir)) 
  dir.create(sh_dir, recursive = TRUE)
write_sh(job_name = 'tabulate-master-prior-break-SELFIES',
         sh_file = sh_file,
         grid_file = grid_file,
         inner_file = 'R/generalization/inner-tabulate-master-break-SELFIES.R',
         system = system,
         time = 6,
         mem = 24)

# finally, run the job on whatever system we're on
submit_job(grid0, sh_file, args$allocation, system)
