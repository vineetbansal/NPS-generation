setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = FALSE)
library(tidyverse)
library(magrittr)
library(data.table)
args = list(); source("R/functions/detect_system.R")

# list files
input_dir = file.path(base_dir, 'prior', 'structural_prior', 'break_SELFIES')
files = list.files(input_dir, recursive = TRUE, full.names = TRUE, 
                   pattern = 'rds')

# set up output directory
output_dir = "data/summary"

# read % ever generated
ever_gen_files = files %>% 
  extract(grepl('ever-generated', .))
ever_gen = map(ever_gen_files, readRDS) %>% 
  map(~ mutate(.x, cv_fold = as.character(cv_fold))) %>% 
  bind_rows()
# save
saveRDS(ever_gen, file.path(output_dir, 'ever-generated-break-SELFIES.rds'))

# read % ever generated
topk_files = files %>% 
  extract(grepl('top-k', .))
topk = map(topk_files, readRDS) %>% 
  map(~ mutate(.x, cv_fold = as.character(cv_fold))) %>% 
  bind_rows()
# save
saveRDS(topk, file.path(output_dir, 'top-k-accuracy-break-SELFIES.rds'))

# read Tc
tc_files = files %>% 
  extract(grepl('Tc', .))
tc = map(tc_files, readRDS) %>% 
  map(~ mutate(.x, cv_fold = as.character(cv_fold))) %>% 
  bind_rows()
# save
saveRDS(tc, file.path(output_dir, 'Tc-break-SELFIES.rds'))

# read number of candidates
num_cand_files = files %>% 
  extract(grepl('num-candidates', .))
num_cands = map(num_cand_files, readRDS) %>% 
  map(~ mutate(.x, cv_fold = as.character(cv_fold))) %>% 
  bind_rows()
# save
saveRDS(num_cands, file.path(output_dir, 'num-candidates-break-SELFIES.rds'))
