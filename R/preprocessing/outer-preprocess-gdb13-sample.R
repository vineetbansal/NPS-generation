setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = FALSE)
library(argparse)

# parse arguments
parser = ArgumentParser(prog = 'outer-preprocess-gdb13-sample.R')
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
  enum_factor = 0
)

# define input file
input_file = file.path(base_dir, 'GDB13', 'gdb13.1M.freq.ll.smi')

# define output files
output_dir = file.path(base_dir, "GDB13", "inputs")
if (!dir.exists(output_dir))
  dir.create(output_dir, recursive = TRUE)
output_filenames =  pmap_chr(grid, function(...) {
  current = tibble(...)
  current %>%
    map2(., names(.), ~ {
      paste0(.y, '=', .x)
    }) %>%
    paste0(collapse = '-') %>%
    paste0('gdb13-1M-', ., '.csv')
})
output_files = file.path(output_dir, output_filenames)

# define vocab files too
vocab_files = gsub("\\.csv", ".vocabulary", output_files)

# now, check for which parameters are already complete
grid0 = grid %>%
  mutate(input_file = input_file,
         output_file = output_files,
         vocab_file = vocab_files) %>%
  filter(!file.exists(output_file) |
           !file.exists(vocab_file)) %>% 
  # shuffle rows
  sample_frac(1)

# write the grid that still needs to be run
grid_file = "sh/grids/preprocess-gdb13-sample.txt"
grid_dir = dirname(grid_file)
if (!dir.exists(grid_dir))
  dir.create(grid_dir, recursive = TRUE)
write.table(grid0, grid_file, quote = FALSE, row.names = FALSE, sep = "\t")

# write the sh file dynamically
sh_file = 'sh/preprocessing/preprocess-gdb13-sample.sh'
sh_dir = dirname(sh_file)
if (!dir.exists(sh_dir)) 
  dir.create(sh_dir, recursive = TRUE)
write_sh(job_name = 'preprocess-gdb13-sample',
         sh_file = sh_file,
         grid_file = grid_file,
         inner_file = 'python/preprocess-gdb13-sample.py',
         system = system,
         time = 24,
         mem = 72)

# finally, run the job on whatever system we're on
submit_job(grid0, sh_file, args$allocation, system)
