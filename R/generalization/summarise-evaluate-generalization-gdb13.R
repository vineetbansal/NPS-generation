setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = F)
library(tidyverse)
library(magrittr)
args = list(); source("R/functions/detect_system.R")

# list files
input_dir = file.path(base_dir, "GDB13", "outcomes")
input_files = list.files(input_dir, pattern = 'rds', full.names = TRUE,
                         recursive = TRUE)

# read data
dats = map(input_files, readRDS)
curves = map(dats, 'curve') %>% 
  setNames(input_files) %>% 
  bind_rows(.id = 'file') %>% 
  mutate(dirname = basename(dirname(file)),
         filename = basename(file)) %>% 
  separate(dirname, into = c('representation', 'enum_factor', 'sample_idx',
                              'rnn_type', 'embedding_size', 'hidden_size',
                              'n_layers', 'dropout', 'batch_size', 
                              'learning_rate'), sep = '-') %>% 
  mutate_at(vars(representation:learning_rate),
            ~ gsub("^.*=|\\.rds$", "", .)) %>% 
  type_convert() %>% 
  dplyr::select(-filename)
summary = map(dats, 'summary') %>% 
  setNames(input_files) %>% 
  bind_rows(.id = 'file') %>% 
  mutate(dirname = basename(dirname(file)),
         filename = basename(file)) %>% 
  separate(dirname, into = c('representation', 'enum_factor', 'sample_idx',
                             'rnn_type', 'embedding_size', 'hidden_size',
                             'n_layers', 'dropout', 'batch_size', 
                             'learning_rate'), sep = '-') %>% 
  mutate_at(vars(representation:learning_rate),
            ~ gsub("^.*=|\\.rds$", "", .)) %>% 
  type_convert() %>% 
  dplyr::select(-filename)

# save
output = list(curves = curves, summary = summary)
output_file = "data/summary/generalization/evaluate-generalization-gdb13.rds"
saveRDS(output, output_file)
