# Create training sets of molecules.
setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = FALSE)
library(argparse)

# parse arguments
parser = ArgumentParser(prog = 'outer-create-training-sets.R')
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
  representation = c('SMILES', 'SELFIES'),
  enum_factor = c(0, 10, 30),
  n_molecules = c(3e4, 1e5, 3e5),
  min_tc = seq(0, 0.15, 0.05),
  sample_idx = seq_len(10)
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

# define output files
output_dir = file.path(base_dir, "training_sets")
if (!dir.exists(output_dir))
  dir.create(output_dir, recursive = TRUE)
output_filenames = pmap_chr(grid, function(...) {
  current = tibble(...)
  current %>%
    map2(., names(.), ~ {
      paste0(.y, '=', .x)
    }) %>%
    paste0(collapse = '-') %>%
    paste0('input-', ., '.smi')
})
output_files = file.path(output_dir, output_filenames)

# define vocabulary files too
vocab_files = gsub("\\.smi$", ".vocab", output_files)

# now, check for which parameters are already complete
grid0 = grid %>%
  mutate(output_file = output_files,
         vocab_file = vocab_files) %>%
  filter(!file.exists(output_file) | 
           !file.exists(vocab_file),
         !file.exists(gsub("\\.smi$", ".err", output_file))) %>% 
  # shuffle rows
  sample_frac(1)

# write the grid that still needs to be run
grid_file = "sh/grids/create-training-sets.txt"
grid_dir = dirname(grid_file)
if (!dir.exists(grid_dir))
  dir.create(grid_dir, recursive = TRUE)
write.table(grid0, grid_file, quote = FALSE, row.names = FALSE, sep = "\t")

# write the sh file dynamically
sh_file = 'sh/preprocessing/create-training-sets.sh'
sh_dir = dirname(sh_file)
if (!dir.exists(sh_dir)) 
  dir.create(sh_dir, recursive = TRUE)
write_sh(job_name = 'create-training-sets',
         sh_file = sh_file,
         grid_file = grid_file,
         inner_file = 'python/inner-create-training-sets.py',
         system = system,
         time = 24,
         mem = 72)

# finally, run the job on whatever system we're on
submit_job(grid0, sh_file, args$allocation, system, job_loop = 10)
