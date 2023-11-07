setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = FALSE)
library(tidyverse)
library(magrittr)
args = list(); source("R/functions/detect_system.R")

###############################################################################-
# train models ####
###############################################################################-

time_files = list.files(file.path(base_dir, 'models'), pattern = 'timing',
                        recursive = TRUE, full.names = TRUE)

# read and combine
times = map(time_files, read.csv)
df = times %>% 
  setNames(basename(dirname(time_files))) %>% 
  bind_rows(.id = 'filename') %>% 
  separate(filename, into = c('database', 'representation', 'enum_factor',
                              'n_molecules', 'min_tc', 'sample_idx', 'rnn_type', 
                              'embedding_size', 'hidden_size', 'n_layers', 
                              'dropout', 'batch_size', 'learning_rate'), 
           sep = '-') %>% 
  mutate_at(vars(database:learning_rate), 
            ~ gsub("^.*=|\\.csv$", "", .)) %>% 
  type_convert()

# save
output_file = "data/summary/timing/train-models.rds"
output_dir = dirname(output_file)
if (!dir.exists(output_dir))
  dir.create(output_dir, recursive = TRUE)
saveRDS(df, output_file)

###############################################################################-
# sample molecules ####
###############################################################################-

time_files = list.files(file.path(base_dir, 'sample'), pattern = 'time-',
                        recursive = TRUE, full.names = TRUE)
# read and combine
times = map(time_files, read.csv)
df = times %>% 
  setNames(basename(dirname(time_files))) %>% 
  bind_rows(.id = 'filename') %>% 
  separate(filename, into = c('database', 'representation', 'enum_factor',
                              'n_molecules', 'min_tc', 'sample_idx', 'rnn_type', 
                              'embedding_size', 'hidden_size', 'n_layers', 
                              'dropout', 'batch_size', 'learning_rate'), 
           sep = '-') %>% 
  mutate_at(vars(database:learning_rate), 
            ~ gsub("^.*=|\\.csv$", "", .)) %>% 
  type_convert()
# save
output_file = "data/summary/timing/sample-models.rds"
output_dir = dirname(output_file)
if (!dir.exists(output_dir))
  dir.create(output_dir, recursive = TRUE)
saveRDS(df, output_file)

###############################################################################-
# parsing valid outputs ####
###############################################################################-

## parsing valid outputs
time_files = list.files(file.path(base_dir, 'sample'), pattern = 'time',
                        recursive = TRUE, full.names = TRUE) %>% 
  extract(!grepl('time-', .))
# read and combine
times = map(time_files, read.csv)
df = times %>% 
  setNames(time_files) %>% 
  bind_rows(.id = 'file') %>% 
  mutate(filename = basename(dirname(file)),
         constraints = ifelse(grepl('constraints', basename(file)),
                              gsub("^.*=|-time\\.csv", "", basename(file)),
                              NA)) %>% 
  separate(filename, into = c('database', 'representation', 'enum_factor',
                              'n_molecules', 'min_tc', 'sample_idx', 'rnn_type', 
                              'embedding_size', 'hidden_size', 'n_layers', 
                              'dropout', 'batch_size', 'learning_rate'), 
           sep = '-') %>% 
  mutate_at(vars(database:learning_rate), 
            ~ gsub("^.*=|\\.csv$", "", .)) %>% 
  type_convert() %>% 
  dplyr::select(-file)
# save
output_file = "data/summary/timing/parse-valid.rds"
output_dir = dirname(output_file)
if (!dir.exists(output_dir))
  dir.create(output_dir, recursive = TRUE)
saveRDS(df, output_file)
