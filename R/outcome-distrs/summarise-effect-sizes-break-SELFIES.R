setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = FALSE)
library(tidyverse)
library(magrittr)
args = list(); source("R/functions/detect_system.R")

# list files
input_dir = file.path(base_dir, "outcome_distrs")
input_files = list.files(input_dir, full.names = TRUE, recursive = TRUE,
                         pattern = 'rds') %>% 
  extract(grepl('constraints', .))

# read and combine
dats = map(input_files, readRDS)
outcome_types = c('atoms', 'murcko', 
                  'discrete', 'continuous')
dfs = map(outcome_types, ~ {
  df = dats %>% 
    map(.x) %>% 
    setNames(input_files) %>% 
    bind_rows(.id = 'file') %>% 
    mutate(dirname = basename(dirname(file)),
           filename = basename(file)) %>% 
    separate(dirname, into = c('database', 'representation', 'enum_factor',
                                'n_molecules', 'min_tc', 'sample_idx', 'rnn_type', 
                                'embedding_size', 'hidden_size', 'n_layers', 
                                'dropout', 'batch_size', 'learning_rate'), 
             sep = '-') %>% 
    separate(filename, into = c('x', 'xx', 'constraints', 'subset'), sep = '-') %>% 
    dplyr::select(-x, -xx, -file) %>% 
    mutate_at(vars(database:learning_rate, constraints, subset), 
              ~ gsub("^.*=|\\.rds$", "", .)) %>% 
    type_convert()
  # discard columns with unique values
  n_uniq = map_int(df, n_distinct)
  df %<>% extract(, n_uniq > 1)
  return(df)
}) %>% setNames(outcome_types)

# save
output_file = "data/summary/effect-sizes-break-SELFIES.rds"
output_dir = dirname(output_file)
if (!dir.exists(output_dir))
  dir.create(output_dir, recursive = TRUE)
saveRDS(dfs, output_file)
