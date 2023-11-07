# Write distributions of physicochemical properties for training molecules.
setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = FALSE)
library(argparse)

# parse arguments
parser = ArgumentParser(prog = 'outer-write-outcome-distrs-training-sets.R')
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
  representation = 'SMILES',
  enum_factor = 0,
  n_molecules = 1e5,
  min_tc = 0,
  sample_idx = seq_len(10)
) %>% 
  mutate(
    # don't put in filename
    max_sampled_mols = n_molecules
  )

# define input files
input_dir = file.path(base_dir, "training_sets")
input_filenames = pmap_chr(grid, function(...) {
  current = tibble(...)
  current %>%
    dplyr::select(database:sample_idx) %>%
    map2(., names(.), ~ {
      paste0(.y, '=', .x)
    }) %>%
    paste0(collapse = '-') %>%
    paste0('input-', ., '.smi')
})
input_files = file.path(input_dir, input_filenames)

# define output files
output_dir = file.path(base_dir, "outcome_distrs", "training")
if (!dir.exists(output_dir)) dir.create(output_dir)
output_dirnames = pmap_chr(grid, function(...) {
  current = tibble(...)
  current %>%
    dplyr::select(database:sample_idx) %>%
    map2(., names(.), ~ {
      paste0(.y, '=', .x)
    }) %>%
    paste0(collapse = '-')
})
output_files = file.path(output_dir, output_dirnames, 'outcome-distrs.csv.gz')

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
sh_file = 'sh/write-outcome-distrs/write-outcome-distrs-training-sets.sh'
sh_dir = dirname(sh_file)
if (!dir.exists(sh_dir)) 
  dir.create(sh_dir, recursive = TRUE)
write_sh(job_name = 'write-outcome-distrs-training-sets',
         sh_file = sh_file,
         grid_file = grid_file,
         inner_file = 'python/inner-write-outcome-distrs.py',
         system = system,
         time = 8,
         mem = 64)

# finally, run the job on whatever system we're on
submit_job(grid0, sh_file, args$allocation, system)
