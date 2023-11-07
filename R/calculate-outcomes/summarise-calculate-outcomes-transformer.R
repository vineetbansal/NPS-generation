# Summarise outcomes from calculate-outcomes-transformer.
setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = F)
library(tidyverse)
library(magrittr)
args = list(); source("R/functions/detect_system.R")

# list check files
output_dir = file.path(base_dir, "outcomes")
output_files = list.files(output_dir, pattern = 'outcomes.csv', 
                          full.names = TRUE, recursive = TRUE) %>% 
  extract(grepl('n_heads', .))

# read and combine
outputs = map(output_files, read.csv)
df = outputs %>% 
  setNames(basename(dirname(output_files))) %>% 
  bind_rows(.id = 'filename') %>% 
  mutate(filename = gsub("5e-04", "0.0005", filename)) %>% 
  separate(filename, into = c('database', 'representation', 'enum_factor',
                              'n_molecules', 'min_tc', 'sample_idx', 
                              'embedding_size', 'n_blocks', 'n_heads', 
                              'max_len', 'dropout', 'exp_factor',
                              'batch_size', 'learning_rate'), 
           sep = '-') %>% 
  mutate_at(vars(database:learning_rate), 
            ~ gsub("^.*=|\\.csv$", "", .)) %>% 
  type_convert() %>% 
  dplyr::select(-input_file)

# discard columns with unique values
n_uniq = map_int(df, n_distinct)
df %<>% extract(, n_uniq > 1)

# save
output_file = "data/summary/calculate-outcomes/calculate-outcomes-transformer.rds"
output_dir = dirname(output_file)
if (!dir.exists(output_dir))
  dir.create(output_dir, recursive = TRUE)
saveRDS(df, output_file)
