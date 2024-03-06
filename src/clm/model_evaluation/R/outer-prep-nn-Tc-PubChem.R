# Prepare a more computationally tractable sample of molecules sampled 
# from the generative model for nearest-neighbor Tc analysis, sampling a
# molecule with a matching molecular formula from PubChem for each.
setwd("~/git/deepmet")
options(stringsAsFactors = FALSE)
library(argparse)

# parse arguments
parser = ArgumentParser(prog = 'outer-prep-nn-Tc-PubChem.R')
parser$add_argument('--allocation', type = 'character', default = "root")
args = parser$parse_args()

library(tidyverse)
library(magrittr)
library(maslib)
detect_system()

# define PubChem file
pubchem_file = file.path(base_dir, 'PubChem/PubChem.tsv')

# set up grid
source("R/functions/grids.R")
grid = final_grid() %>% 
  tidyr::crossing(sample_idx = 1)

# define model sample files
sample_dir = file.path(base_dir, "CV", "sample")
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
                                    paste0('unique-masses-novel-', 
                                           ceiling(sample_idx / 10), 
                                           '.tsv.gz')))

# create a new grid from just training and sample files
grid = data.frame(sample_file = sample_files) %>% 
  distinct() %>% 
  # add PubChem
  mutate(pubchem_file = pubchem_file,
         match_on = 'formula')

# define output files
output_dir = file.path(base_dir, "CV", "nn_Tc", "PubChem")
output_dirnames = basename(dirname(grid$sample_file))
output_dirs = file.path(output_dir, output_dirnames)
output_files = file.path(output_dirs, 'nn-Tc-input.csv')

# now, check for which parameters are already complete
grid0 = grid %>%
  mutate(output_file = output_files) %>% 
  filter(file.exists(sample_file),
         !file.exists(output_file))

# write the grid that still needs to be run
grid_file = "sh/grids/CV/prep-nn-Tc-PubChem.txt"
grid_dir = dirname(grid_file)
if (!dir.exists(grid_dir))
  dir.create(grid_dir, recursive = TRUE)
write.table(grid0, grid_file, quote = FALSE, row.names = FALSE, sep = "\t")

# write the sh file dynamically
sh_file = '~/git/deepmet/sh/CV/prep-nn-Tc-PubChem.sh'
sh_dir = dirname(sh_file)
if (!dir.exists(sh_dir)) 
  dir.create(sh_dir, recursive = TRUE)
write_sh(job_name = 'prep-Tc-PubChem',
         sh_file = sh_file,
         grid_file = grid_file,
         inner_file = 'R/nn-Tc/inner-prep-nn-Tc-PubChem.R',
         time = 24,
         mem = 64,
         env = '/scratch/st-ljfoster-1/decoy-generation/env-r')

# finally, run the job on whatever system we're on
submit_job(grid0, sh_file, allocation = allocation)

