setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = FALSE)
library(argparse)

# parse arguments
parser = ArgumentParser(prog = 'outer-preprocess-prior-datasets.R')
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
  cv_fold = seq_len(10)
)

# define input files
input_files = with(grid, paste0("data/prior/raw/", database, ".txt")) %>% 
  path.expand()

# define output files
output_dir = file.path(base_dir, "prior", "inputs")
if (!dir.exists(output_dir))
  dir.create(output_dir, recursive = TRUE)
output_filenames = pmap_chr(grid, function(...) {
  current = tibble(...)
  current %>%
    map2(., names(.), ~ {
      paste0(.y, '=', .x)
    }) %>%
    paste0(collapse = '-') %>%
    paste0('train-', ., '.csv')
})
train_files = file.path(output_dir, output_filenames)
test_files = gsub("train-", "test-", train_files)

# define vocab files too
vocab_files = gsub("\\.csv", ".vocabulary", train_files)

# now, check for which parameters are already complete
grid0 = grid %>%
  mutate(input_file = input_files,
         train_file = train_files,
         test_file = test_files,
         vocab_file = vocab_files) %>%
  filter(!file.exists(train_file) | 
           !file.exists(test_file) |
           !file.exists(vocab_file)) %>% 
  # shuffle rows
  sample_frac(1)

# write the grid that still needs to be run
grid_file = "sh/grids/preprocess-prior-datasets.txt"
grid_dir = dirname(grid_file)
if (!dir.exists(grid_dir))
  dir.create(grid_dir, recursive = TRUE)
write.table(grid0, grid_file, quote = FALSE, row.names = FALSE, sep = "\t")

# write the sh file dynamically
sh_file = 'sh/preprocessing/preprocess-prior-datasets.sh'
sh_dir = dirname(sh_file)
if (!dir.exists(sh_dir)) 
  dir.create(sh_dir, recursive = TRUE)
write_sh(job_name = 'preprocess-prior-datasets',
         sh_file = sh_file,
         grid_file = grid_file,
         inner_file = 'python/inner-preprocess-prior-datasets.py',
         system = system,
         time = 24,
         mem = 72)

# finally, run the job on whatever system we're on
submit_job(grid0, sh_file, args$allocation, system)
