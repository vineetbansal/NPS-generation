# Summarise outcomes from calculate-outcomes-RNN.
setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = F)
library(tidyverse)
library(magrittr)
args = list(); source("R/functions/detect_system.R")

# list check files
output_dir = file.path(base_dir, "outcomes")
output_files = list.files(output_dir, pattern = 'outcomes.csv', 
                          full.names = TRUE, recursive = TRUE)

# read and combine
outputs = map(output_files, read.csv)
df = outputs %>% 
  setNames(basename(dirname(output_files))) %>% 
  bind_rows(.id = 'filename') %>% 
  separate(filename, into = c('database', 'representation', 'enum_factor',
                              'n_molecules', 'min_tc', 'sample_idx', 'rnn_type', 
                              'embedding_size', 'hidden_size', 'n_layers', 
                              'dropout', 'batch_size', 'learning_rate'), 
           sep = '-') %>% 
  mutate_at(vars(database:learning_rate), 
            ~ gsub("^.*=|\\.csv$", "", .)) %>% 
  type_convert()

# discard columns with unique values
n_uniq = map_int(df, n_distinct)
df %<>% extract(, n_uniq > 1)
# remove input file
df %<>% dplyr::select(-input_file)

# save
output_file = "data/summary/calculate-outcomes/calculate-outcomes-RNN.rds"
output_dir = dirname(output_file)
if (!dir.exists(output_dir))
  dir.create(output_dir, recursive = TRUE)
saveRDS(df, output_file)
