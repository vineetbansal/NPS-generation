setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = FALSE)
library(tidyverse)
library(magrittr)
library(data.table)
args = list(); source("R/functions/detect_system.R")

# suppress summarise info
options(dplyr.summarise.inform = FALSE)

# list files
input_dir = file.path(base_dir, 'prior', 'structural_prior')
input_files = list.files(input_dir, recursive = TRUE, full.names = TRUE,
                         pattern = 'CV-Tc')

# create metadata
meta = data.frame(file = input_files,
                  filename = basename(dirname(input_files))) %>% 
  separate(filename, into = c('database', 'representation', 'enum_factor',
                              'sample_idx', 'k', 'cv_fold', 'rnn_type',
                              'embedding_size', 'hidden_size', 'n_layers',
                              'dropout', 'batch_size', 'learning_rate'),
           sep = '-') %>% 
  mutate_at(vars(database:learning_rate), ~ gsub("^.*=", "", .)) %>% 
  type_convert() %>% 
  mutate(representation = ifelse(grepl('break_SELFIES', file),
                                 'Unconstrained SELFIES', representation)) %>% 
  # remove fixed variables
  extract(, map_lgl(., ~ n_distinct(.x) > 1))

# read all files
dats = map(meta$file, ~ fread(.x) %>% 
             mutate(target_source = ifelse(target_source == "model" & 
                                             target_rank > 0, 
                                           "model (random)", target_source)))

# merge in metadata and combine per database
for (database in unique(meta$database)) {
  database_idxs = which(meta$database == database)
  meta0 = meta %>% extract(database_idxs, )
  dats0 = dats %>% extract(database_idxs)
  df = map_dfr(seq_len(nrow(meta0)), ~ cbind(meta0[.x, ], dats0[[.x]])) %>% 
    dplyr::select(-file, -Index) %>% 
    drop_na(Tc) %>% 
    filter(!(grepl('SELFIES', representation) & 
               target_source %in% c('model (random)', 'PubChem', 'train')))
 
  # save
  output_dir = file.path(base_dir, 'prior', 'structural_prior')
  saveRDS(df, file.path(output_dir, paste0('Tc-all-', database, '.rds')))
}
