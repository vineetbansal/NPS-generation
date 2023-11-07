# Combine samples from transformer models, which were drawn 1m at a time.
setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = FALSE)
library(argparse)

# parse arguments
parser = ArgumentParser(prog = 'outer-consolidate-transformer-samples.R')
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
  representation = c('SMILES', 'SELFIES'),
  enum_factor = 0,
  n_molecules = 1e5,
  min_tc = 0,
  sample_idx = seq_len(10),
  
  # architecture - fixed
  embedding_size = 256,
  n_blocks = 8,
  n_heads = 8,
  max_len = 250,
  dropout = 0,
  exp_factor = 4,
  batch_size = 64,
  learning_rate = 0.0005,
  
  ## don't put in filename
  mol_sample_idx = '$', ## placeholder
  max_mols = 10e6
)

# define input files (*vocab)
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
vocab_files = gsub("\\.smi$", ".vocab", input_files)

# define sample output files
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

# define output files
output_files = with(grid, file.path(sample_dir, sample_dirnames,
                                    'sample-0.csv'))

# now, check for which parameters are already complete
grid0 = grid %>%
  mutate(vocab_file = vocab_files,
         input_file = sample_files,
         check_file = check_files,
         output_file = output_files) %>% 
  mutate(input_ok = map_lgl(input_file, ~ { 
    input_file = .x
    map_lgl(seq_len(10), ~ gsub('\\$', .x, input_file) %>% file.exists()) %>%
      all()
    }),
    check_ok = map_lgl(check_files, ~ { 
      check_files = .x
      map_lgl(seq_len(10), ~ gsub('\\$', .x, check_files) %>% file.exists()) %>%
        all()
    })) %>% 
  filter(file.exists(vocab_file),
         input_ok,
         check_ok,
         !file.exists(output_file))
# remove columns that are all NA
keep = map_lgl(grid0, ~ any(!is.na(.x) & .x != 'NA'))
grid0 %<>% extract(, keep)

# write the grid that still needs to be run
grid_file = "sh/grids/consolidate-transformer-samples.txt"
grid_dir = dirname(grid_file)
if (!dir.exists(grid_dir))
  dir.create(grid_dir, recursive = TRUE)
write.table(grid0, grid_file, quote = FALSE, row.names = FALSE, sep = "\t")

# write the sh file dynamically
sh_file = '~sh/sample-molecules/consolidate-transformer-samples.sh'
sh_dir = dirname(sh_file)
if (!dir.exists(sh_dir)) 
  dir.create(sh_dir, recursive = TRUE)
write_sh(job_name = 'consolidate-transformer-samples',
         sh_file = sh_file,
         grid_file = grid_file,
         inner_file = 'R/sample-molecules/inner-consolidate-transformer-samples.R',
         system = system,
         time = 24,
         mem = 24)

# finally, run the job on whatever system we're on
submit_job(grid0, sh_file, args$allocation, system)
