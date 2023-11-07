setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = FALSE)
library(argparse)

# dynamically parse arguments
parser = ArgumentParser(prog = 'inner-calculate-effect-sizes.R')
grid = read.delim("sh/grids/calculate-effect-sizes.txt")
for (param_name in colnames(grid))
  parser$add_argument(paste0('--', param_name),
                      type = typeof(grid[[param_name]]))
args = parser$parse_args()
print(args)

library(tidyverse)
library(magrittr)
library(effsize)
library(broom)
source("R/theme.R")

# files must be gzipped
filetype = function(path) {
  f = file(path)
  ext = summary(f)$class
  close.connection(f)
  ext
}
if (filetype(args$train_file) != 'gzfile' |
    filetype(args$sample_file) != 'gzfile')
  stop('file(s) not gzipped')

# read files
train = read_csv(args$train_file) %>% 
  mutate(xval = '0-Training set')
sample = read_csv(args$sample_file) %>% 
  mutate(xval = '1-Sampled molecules')

# combine
dat = bind_rows(train, sample)

###############################################################################-
# Atom counts ####
###############################################################################-

# extract atom counts
counts = filter(dat, grepl('\\# atoms,', outcome)) %>% 
  # convert to proportions
  group_by(xval) %>% 
  mutate(prop = value / sum(value)) %>% 
  ungroup() %>% 
  # extract atom symbols
  mutate(symbol = gsub("^.* ", "", outcome) %>% 
           # relevel
           fct_relevel('C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I') %>% 
           fct_rev())

# calculate odds ratios relative to training set
atoms = unique(counts$symbol)
atom_ORs = map_dfr(atoms, ~ {
  atom = as.character(.x)
  print(atom)
  ref = filter(counts, xval =='0-Training set') %>% 
    mutate(y = ifelse(symbol == atom, atom, 'other') %>% 
             fct_relevel(atom, 'other')) %>% 
    dplyr::count(y, wt = value)
  obs = filter(counts, xval == '1-Sampled molecules') %>% 
    mutate(y = ifelse(symbol == atom, atom, 'other') %>% 
             fct_relevel(atom, 'other')) %>% 
    dplyr::count(y, wt = value)
  test = tidy(fisher.test(cbind(ref$n, obs$n))) %>% 
    mutate(symbol = atom,
           freq_ref = ref$n[1] / sum(ref$n),
           freq_obs = obs$n[1] / sum(obs$n))
})

###############################################################################-
# Discrete outcomes ####
###############################################################################-

# iterate through discrete outcoems
discrete_outcomes = c('# of rings',
                      '# of aliphatic rings',
                      '# of aromatic rings',
                      '# of hydrogen donors',
                      '# of hydrogen acceptors')
discrete = map_dfr(discrete_outcomes, ~ {
  discrete_outcome = .x
  message("working on discrete outcome: ", discrete_outcome, " ...")
  
  # subset data
  dat0 = filter(dat, outcome == discrete_outcome) %>% 
    # winsorize
    mutate(value = ifelse(value > 20, 20, value))
  
  # convert to proportions
  counts = dplyr::count(dat0, xval, value) %>% 
    # convert to proportions
    group_by(xval) %>% 
    mutate(prop = n / sum(n)) %>% 
    ungroup() %>% 
    mutate(value = factor(value, levels = seq(0, 20)) %>% 
             fct_rev())
  
  # calculate odds ratios relative to training set
  discrete_values = unique(counts$value)
  discrete_ORs = map_dfr(discrete_values, ~ {
    discrete_value = as.character(.x)
    print(discrete_value)
    ref = filter(counts, xval == '0-Training set') %>% 
      mutate(y = ifelse(value == discrete_value, discrete_value, 'other') %>% 
               fct_relevel(discrete_value, 'other')) %>% 
      dplyr::count(y, wt = n)
    obs = filter(counts, xval == '1-Sampled molecules') %>% 
      mutate(y = ifelse(value == discrete_value, discrete_value, 'other') %>% 
               fct_relevel(discrete_value, 'other')) %>% 
      dplyr::count(y, wt = n)
    test = tidy(fisher.test(cbind(ref$n, obs$n))) %>% 
      mutate(value = discrete_value,
             freq_ref = ref$n[1] / sum(ref$n),
             freq_obs = obs$n[1] / sum(obs$n))
  }) %>% 
    mutate(outcome = discrete_outcome) %>% 
    dplyr::select(outcome, everything())
})

###############################################################################-
# Continuous outcomes ####
###############################################################################-

# iterate through continuous outcomes
continuous_outcomes = unique(dat$outcome) %>% 
  extract(!grepl(' atoms|Murcko| of rings', .))
continuous = map_dfr(continuous_outcomes, ~ {
  continuous_outcome = .x
  message("working on continuous outcome: ", continuous_outcome, " ...")
  dat0 = filter(dat, outcome == continuous_outcome) %>% 
    mutate(xval_f = factor(xval, levels = c('0-Training set', '1-Sampled molecules')))
  eff = cohen.d(d = dat0$value, f = dat0$xval_f)$estimate
  summary = dat0 %>% 
    group_by(xval) %>% 
    do(tidy(summary(.$value))) %>%
    mutate(eff = eff) %>% 
    mutate(outcome = continuous_outcome) %>% 
    dplyr::select(outcome, everything())
})

###############################################################################-
# Combine and save ####
###############################################################################-

output = list(atoms = atom_ORs,
              discrete = discrete, 
              continuous = continuous)
output_dir = dirname(args$output_file)
if (!dir.exists(output_dir))
  dir.create(output_dir, recursive = TRUE)
saveRDS(output, args$output_file)
