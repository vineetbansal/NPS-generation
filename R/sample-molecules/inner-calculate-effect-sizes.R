setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = FALSE)
library(argparse)

# dynamically parse arguments
parser = ArgumentParser(prog = 'inner-calculate-effect-sizes.R')
grid = read.delim("sh/grids/plot-calculate-effect-sizes.txt")
for (param_name in colnames(grid))
  parser$add_argument(paste0('--', param_name),
                      type = typeof(grid[[param_name]]))
args = parser$parse_args()
print(args)

library(tidyverse)
library(magrittr)
library(effsize)
source("R/theme.R")

# read data
dat = read_csv(args$valid_file, col_names = c('loss', 'smiles', 'tokens', 'msg', 
                                              'error')) %>% 
  # tag valid vs. not
  mutate(valid = msg == 'valid') %>% 
  # filter empty samples
  filter(tokens > 0)

# calculate effect sizes
dat %<>% mutate(valid_f = factor(as.character(valid), levels = c('FALSE', 'TRUE')))
eff1 = cohen.d(d = dat$loss, f = dat$valid_f)$estimate
errors = unique(na.omit(dat$error))
eff2s = map_dbl(errors, ~ {
  dat0 = filter(dat, is.na(error) | error == .x)
  cohen.d(d = dat0$loss, f = dat0$valid_f)$estimate
}) %>% setNames(errors)

# create data frame
eff = data.frame(
  analysis = c('loss',
               paste0('loss: ', names(eff2s))),
  d = c(eff1, eff2s)
)
# write
write.csv(eff, args$effect_file, row.names = FALSE)

# save a table with the total proportion of each error
freqs = dplyr::count(dat, error, valid) %>% 
  mutate(proportion = n / sum(n))
write.csv(freqs, args$freq_file, row.names = FALSE)
