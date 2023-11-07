setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = FALSE)
library(argparse)

# dynamically parse arguments
parser = ArgumentParser(prog = 'inner-summarise-formula-prior.R')
grid = read.delim("sh/grids/summarise-formula-prior.txt")
for (param_name in colnames(grid))
  parser$add_argument(paste0('--', param_name),
                      type = typeof(grid[[param_name]]))
args = parser$parse_args()
print(args)

library(tidyverse)
library(magrittr)
library(data.table)
# suppress summarise info
options(dplyr.summarise.inform = FALSE)

# set up input files
cv_folds = seq_len(10)
rank_files = map_chr(cv_folds, ~ gsub('\\$', .x, args$rank_file))

# set up output directory
output_dir = dirname(args$rank_file)
if (!dir.exists(output_dir))
  dir.create(output_dir, recursive = TRUE)

# read ranks
ranks = map(rank_files, fread)

# total number of formulas
n_formulas = ranks %>% 
  map(~ .x %>% 
        dplyr::select(target_source, n_candidates)) %>% 
  setNames(basename(dirname(rank_files))) %>% 
  bind_rows(.id = 'file') %>% 
  separate(file, into = c('database', 'representation', 'enum_factor',
                          'sample_idx', 'k', 'cv_fold',
                          'rnn_type', 'embedding_size', 'hidden_size', 
                          'n_layers', 'dropout', 'batch_size', 'learning_rate'),
           sep = '-') %>% 
  mutate_at(vars(database:learning_rate), 
            ~ gsub("^.*=|/.*$", "", .)) %>% 
  type_convert()

#' % ever generated
ever_gen = ranks %>% 
  map(~ .x %>% 
        group_by(target_source) %>% 
        summarise(ever_gen = mean(!is.na(target_rank)),
                  n = n()) %>% 
        ungroup()) %>% 
  setNames(basename(dirname(rank_files))) %>% 
  bind_rows(.id = 'file') %>% 
  separate(file, into = c('database', 'representation', 'enum_factor',
                          'sample_idx', 'k', 'cv_fold',
                          'rnn_type', 'embedding_size', 'hidden_size', 
                          'n_layers', 'dropout', 'batch_size', 'learning_rate'),
           sep = '-') %>% 
  mutate_at(vars(database:learning_rate), 
            ~ gsub("^.*=|/.*$", "", .)) %>% 
  type_convert()

# extract max rank for CDFs
max_rank = map_dbl(ranks, ~ max(.$target_rank, na.rm = TRUE)) %>% max()

#' top-k accuracy
topk = ranks %>% 
  map(~ { 
    df = .x
    rnks = seq_len(1e3)
    topk = map_dfr(rnks, ~ {
      rank = .x
      df %>% 
        group_by(target_source) %>% 
        summarise(top_k = mean(!is.na(target_rank) & target_rank < rank),
                  n = n()) %>% 
        ungroup() %>% 
        mutate(rank = rank)
    })
  }) %>% 
  setNames(basename(dirname(rank_files))) %>% 
  bind_rows(.id = 'file') %>% 
  separate(file, into = c('database', 'representation', 'enum_factor',
                          'sample_idx', 'k', 'cv_fold',
                          'rnn_type', 'embedding_size', 'hidden_size', 
                          'n_layers', 'dropout', 'batch_size', 'learning_rate'),
           sep = '-') %>% 
  mutate_at(vars(database:learning_rate), ~ gsub("^.*=|/.*$", "", .)) %>% 
  type_convert()

# save all files
saveRDS(n_formulas, file.path(output_dir, 'num-formulas.rds'))
saveRDS(ever_gen, file.path(output_dir, 'ever-generated.rds'))
saveRDS(topk, file.path(output_dir, 'top-k-accuracy.rds'))
