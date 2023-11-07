setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = FALSE)
library(tidyverse)
library(magrittr)
library(data.table)
args = list(); source("R/functions/detect_system.R")

# list files
input_dir = file.path(base_dir, 'prior', 'formula_prior')
files = list.files(input_dir, recursive = TRUE, full.names = TRUE, 
                   pattern = 'rds') %>% 
  extract(!grepl('break_SELFIES', .))

# set up output directory
output_dir = "data/summary"

# read % ever generated
ever_gen_files = files %>% 
  extract(grepl('ever-generated', .))
ever_gen = map(ever_gen_files, readRDS) %>% 
  map(~ mutate(.x, cv_fold = as.character(cv_fold))) %>% 
  bind_rows()
# save
saveRDS(ever_gen, file.path(output_dir, 'ever-generated-formula.rds'))

# read % ever generated
topk_files = files %>% 
  extract(grepl('top-k', .))
topk = map(topk_files, readRDS) %>% 
  map(~ mutate(.x, cv_fold = as.character(cv_fold))) %>% 
  bind_rows()
# save
saveRDS(topk, file.path(output_dir, 'top-k-accuracy-formula.rds'))

# read number of formulas
num_formula_files = files %>% 
  extract(grepl('num-formulas', .))
num_formulas = map(num_formula_files, readRDS) %>% 
  map(~ mutate(.x, cv_fold = as.character(cv_fold))) %>% 
  bind_rows()
# save
saveRDS(num_formulas, file.path(output_dir, 'num-formulas.rds'))
