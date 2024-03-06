setwd("~/git/deepmet")
options(stringsAsFactors = FALSE)
library(argparse)

# dynamically parse arguments
parser = ArgumentParser(prog = 'inner-prep-nn-Tc-PubChem.R')
grid = read.delim("sh/grids/CV/prep-nn-Tc-PubChem.txt")
for (param_name in colnames(grid))
  parser$add_argument(paste0('--', param_name),
                      type = typeof(grid[[param_name]]))
args = parser$parse_args()
print(args)

library(tidyverse)
library(magrittr)
library(data.table)

# read the tabulated data
dat = fread(args$sample_file, data.table = FALSE)

# sample a subset of 500,000 molecules, weighted by frequency
n_tries = 500e3
idxs = sample(seq_len(nrow(dat)), size = n_tries, replace = TRUE, 
              prob = dat$size)
sample = dat[idxs, ]

# read PubChem
pubchem = fread(args$pubchem_file, data.table = FALSE, header = FALSE,
                col.names = c('smiles', 'mass', 'formula'))

# filter PubChem to relevant formulas
fmlas = unique(sample$formula)
pubchem %<>% filter(formula %in% fmlas)

# split by formula
split = pubchem %>% split(.$formula)

# for each molecule, try to find a match from PubChem
matches = pmap_dfr(sample, function(...) {
  current = tibble(...)
  cands = split[[current$formula]]
  if (!current$formula %in% names(split) || nrow(cands) == 0) {
    return(current %>% mutate(source = 'DeepMet'))
  } else {
    match = sample_n(cands, 1)
    ## don't re-sample this molecule again
    # cands = anti_join(cands, match)
    # split[[current$formula]] <<- cands
    ## ^ ignored - sample with replacement
  }
  bind_rows(current %>% mutate(source = 'DeepMet'),
            match %>% mutate(source = 'PubChem'))
})

# write
output_dir = dirname(args$output_file)
if (!dir.exists(output_dir))
  dir.create(output_dir, recursive = TRUE)
write.csv(matches, args$output_file, row.names = FALSE)
