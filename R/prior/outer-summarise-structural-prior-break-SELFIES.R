setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = FALSE)
library(argparse)

# parse arguments
parser = ArgumentParser(prog = 'outer-summarise-structural-prior-break-SELFIES.R')
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
  cv_fold = '$', # placeholder
  
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

# define output directories
output_dir = file.path(base_dir, "prior", "structural_prior", "break_SELFIES")
output_dirnames = pmap_chr(grid, function(...) {
  current = tibble(...)
  current %>%
    dplyr::select(database:learning_rate) %>%
    map2(., names(.), ~ {
      paste0(.y, '=', .x)
    }) %>%
    paste0(collapse = '-')
})
output_dirs = file.path(output_dir, output_dirnames)
output_files = file.path(output_dirs, 'Tc.rds')

# define rank files
rank_files = file.path(output_dirs, 'CV-ranks.csv.gz')
# define Tc files
tc_files = file.path(output_dirs, 'CV-Tc.csv.gz')

# now, check for which parameters are already complete
grid0 = grid %>%
  mutate(output_dir = output_dirs,
         output_file = output_files,
         rank_file = rank_files,
         tc_file = tc_files) %>% 
  mutate(
    rank_exists = map_lgl(rank_file, ~ {
      file = .x
      fold_idxs = seq_len(10)
      files = map_chr(fold_idxs, ~ gsub('\\$', .x, file))
      all(file.exists(files))
    }),
    tc_exists = map_lgl(tc_file, ~ {
      file = .x
      fold_idxs = seq_len(10)
      files = map_chr(fold_idxs, ~ gsub('\\$', .x, file))
      all(file.exists(files))
    })
  ) %>% 
  filter(rank_exists,
         tc_exists,
         !file.exists(output_file))

# write the grid that still needs to be run
grid_file = "sh/grids/summarise-structural-prior.txt"
grid_dir = dirname(grid_file)
if (!dir.exists(grid_dir))
  dir.create(grid_dir, recursive = TRUE)
write.table(grid0, grid_file, quote = FALSE, row.names = FALSE, sep = "\t")

# write the sh file dynamically
sh_file = 'sh/summarise-structural-prior.sh'
sh_dir = dirname(sh_file)
if (!dir.exists(sh_dir)) 
  dir.create(sh_dir, recursive = TRUE)
write_sh(job_name = 'summarise-prior',
         sh_file = sh_file,
         grid_file = grid_file,
         inner_file = 'R/prior/inner-summarise-structural-prior.R',
         system = system,
         time = 8,
         mem = 48)

# finally, run the job on whatever system we're on
submit_job(grid0, sh_file, args$allocation, system)
