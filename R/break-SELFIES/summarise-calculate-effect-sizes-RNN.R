setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = F)
library(tidyverse)
library(magrittr)
args = list(); source("R/functions/detect_system.R")

# list check files
output_dir = file.path(base_dir, 'parse-valid-selfies')
output_files = list.files(output_dir, pattern = 'effect.csv', full.names = TRUE,
                          recursive = TRUE)

# read and combine
dats = map(output_files, read.csv) 
dat = dats %>% 
  setNames(basename(output_files)) %>% 
  bind_rows(.id = 'filename') %>% 
  separate(filename, into = c('dirname', 'filename'), sep = '__') %>% 
  separate(dirname, into = c('database', 'representation', 'enum_factor',
                             'n_molecules', 'min_tc', 'sample_idx', 'rnn_type', 
                             'embedding_size', 'hidden_size', 'n_layers', 
                             'dropout', 'batch_size', 'learning_rate'), 
           sep = '-') %>% 
  mutate_at(vars(database:learning_rate), 
            ~ gsub("^.*=|\\.csv$", "", .)) %>% 
  type_convert() %>% 
  # extract contraints
  mutate(constraints = gsub("^.*constraints=|-valid.*$", "", filename)) %>% 
  dplyr::select(-filename)

# do the same for frequency files
freq_files = list.files(output_dir, pattern = 'freqs.csv', full.names = TRUE,
                        recursive = TRUE)
freqs = map(freq_files, read.csv) 
freq = freqs %>% 
  setNames(basename(output_files)) %>% 
  bind_rows(.id = 'filename') %>% 
  separate(filename, into = c('dirname', 'filename'), sep = '__') %>% 
  separate(dirname, into = c('database', 'representation', 'enum_factor',
                             'n_molecules', 'min_tc', 'sample_idx', 'rnn_type', 
                             'embedding_size', 'hidden_size', 'n_layers', 
                             'dropout', 'batch_size', 'learning_rate'), 
           sep = '-') %>% 
  mutate_at(vars(database:learning_rate), 
            ~ gsub("^.*=|\\.csv$", "", .)) %>% 
  type_convert() %>% 
  # extract contraints
  mutate(constraints = gsub("^.*constraints=|-valid.*$", "", filename)) %>% 
  dplyr::select(-filename)

# save both
output = list(effects = dat, freqs = freq)
output_file = "data/summary/break-SELFIES/parse-valid-SELFIES-RNN.rds"
output_dir = dirname(output_file)
if (!dir.exists(output_dir))
  dir.create(output_dir, recursive = TRUE)
saveRDS(output, output_file)
