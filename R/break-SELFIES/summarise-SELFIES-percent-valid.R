# Write the total proportion of valid SELFIES when parsed without constraints. 
setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = FALSE)
library(tidyverse)
library(magrittr)
args = list(); source("R/functions/detect_system.R")

# list all files
input_dir = file.path(base_dir, "sample")
input_files = list.files(input_dir, pattern = "constraints", 
                         recursive = TRUE, full.names = TRUE) %>% 
  extract(grepl('-valid\\.csv', .)) %>% 
  extract(!grepl('\\.gz$', .))

# count lines
lines = map_chr(input_files, ~ system(paste('wc -l', .x), intern = TRUE)) %>% 
  gsub(" .*$", "", .) %>% 
  as.integer()

# divide by total sample size
valid = lines / 1e7

# combine
df = data.frame(dirname = basename(dirname(input_files)),
                filename = basename(input_files),
                lines = lines,
                valid = valid) %>% 
  separate(dirname, into = c('database', 'representation', 'enum_factor',
                              'n_molecules', 'min_tc', 'sample_idx', 'rnn_type', 
                              'embedding_size', 'hidden_size', 'n_layers', 
                              'dropout', 'batch_size', 'learning_rate'), 
           sep = '-') %>% 
  mutate_at(vars(database:learning_rate), 
            ~ gsub("^.*=|\\.csv$", "", .)) %>% 
  type_convert() %>% 
  # flag constraints
  mutate(constraints = gsub("^.*constraints=|-valid.*$", "", filename)) %>% 
  dplyr::select(-filename)

# discard columns with unique values
n_uniq = map_int(df, n_distinct)
df %<>% extract(, n_uniq > 1)

# save
output_file = "data/summary/break-SELFIES/SELFIES-percent-valid.rds"
output_dir = dirname(output_file)
if (!dir.exists(output_dir))
  dir.create(output_dir, recursive = TRUE)
saveRDS(df, output_file)
