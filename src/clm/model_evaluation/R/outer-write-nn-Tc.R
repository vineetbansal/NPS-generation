setwd("~/git/deepmet")
options(stringsAsFactors = FALSE)
library(argparse)

# parse arguments
parser = ArgumentParser(prog = 'outer-write-nn-Tc.R')
parser$add_argument('--allocation', type = 'character', default = "root")
args = parser$parse_args()

library(tidyverse)
library(magrittr)
library(maslib)
detect_system()

# set up grid
source("R/functions/grids.R")
grid = final_grid() %>% 
  mutate(sample_idx = 1)

# define input files (samples from each CV fold of the model)
input_dir = file.path(base_dir, "CV", "sample")
input_dirnames = pmap_chr(grid, function(...) {
  current = tibble(...)
  current %>%
    dplyr::select(database:learning_rate) %>%
    map2(., names(.), ~ {
      paste0(.y, '=', .x)
    }) %>%
    paste0(collapse = '-')
})
input_files = with(grid, file.path(input_dir, input_dirnames,
                                    paste0('unique-masses-novel-', 
                                           sample_idx, '.tsv.gz')))

# define training files (target nn's for each fold)
train_dir = file.path(base_dir, "CV", "inputs")
train_filenames = pmap_chr(grid, function(...) {
  current = tibble(...)
  current %>%
    mutate(enum_factor = 0) %>% 
    dplyr::select(database, enum_factor, downsample_nonlipids, downsample_n,
                  include_lipids, k, cv_fold) %>% 
    map2(., names(.), ~ {
      paste0(.y, '=', .x)
    }) %>%
    paste0(collapse = '-') %>%
    paste0('train-', ., '.csv')
})
train_files = file.path(train_dir, train_filenames)

# create a new grid from just training and sample files
grid = data.frame(input_file = input_files,
                  reference_file = train_files) %>% 
  distinct()

# now, define the actual input files
## set 1: by frequency
input_dir1 = file.path(base_dir, "CV", "nn_Tc", "freq")
input_dirnames1 = basename(dirname(grid$input_file))
input_files1 = file.path(input_dir1, input_dirnames1, 'nn-Tc-input.csv')
## set 2: vs PubChem
input_dir2 = file.path(base_dir, "CV", "nn_Tc", "PubChem")
input_dirnames2 = basename(dirname(grid$input_file))
input_files2 = file.path(input_dir2, input_dirnames2, 'nn-Tc-input.csv')

# re-create the grid
grid0 = bind_rows(
  # frequency
  data.frame(query_file = input_files1, reference_file = grid$reference_file),
  # PubChem
  data.frame(query_file = input_files2, reference_file = grid$reference_file)
) %>% 
  # add output files
  mutate(output_file = gsub("-input\\.csv$", ".csv", query_file)) %>% 
  # now, check for which parameters are already complete
  filter(file.exists(query_file),
         file.exists(reference_file),
         !file.exists(output_file))

# write the grid that still needs to be run
grid_file = "sh/grids/CV/write-nn-Tc.txt"
grid_dir = dirname(grid_file)
if (!dir.exists(grid_dir))
  dir.create(grid_dir, recursive = TRUE)
write.table(grid0, grid_file, quote = FALSE, row.names = FALSE, sep = "\t")

# write the sh file dynamically
sh_file = '~/git/deepmet/sh/CV/write-nn-Tc.sh'
sh_dir = dirname(sh_file)
if (!dir.exists(sh_dir)) 
  dir.create(sh_dir, recursive = TRUE)
write_sh(job_name = 'write-nn-Tc',
         sh_file = sh_file,
         grid_file = grid_file,
         inner_file = 'python/write-nn-Tc.py',
         time = 24,
         mem = 64,
         env = '/scratch/st-ljfoster-1/decoy-generation/env')

# finally, run the job on whatever system we're on
submit_job(grid0, sh_file, allocation = allocation)
