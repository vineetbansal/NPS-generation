setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = FALSE)
library(argparse)

# dynamically parse arguments
parser = ArgumentParser(prog = 'inner-write-valid-by-decile.R')
grid = read.delim("sh/grids/write-valid-by-decile.txt")
for (param_name in colnames(grid))
  parser$add_argument(paste0('--', param_name),
                      type = typeof(grid[[param_name]]))
args = parser$parse_args()
print(args)

library(tidyverse)
library(magrittr)

# read data
dat = read_csv(args$valid_file, col_names = c('loss', 'smiles', 'tokens', 'msg', 
                                              'error')) %>% 
  # tag valid vs. not
  mutate(valid = msg == 'valid') %>% 
  # filter empty samples
  filter(tokens > 0)

# calculate deciles
summary = dat %>% 
  mutate(decile = ntile(loss, n = 10)) %>% 
  group_by(decile) %>% 
  summarise(mean = mean(loss), median = median(loss), valid = mean(valid)) %>% 
  ungroup() %>% 
  mutate(outcome = 'loss')

# save
output_dir = dirname(args$output_file)
if (!dir.exists(output_dir))
  dir.create(output_dir, recursive = TRUE)
write.csv(summary, args$output_file, row.names = FALSE)
