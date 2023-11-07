setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = F)
library(tidyverse)
library(magrittr)
library(broom)
source("R/theme.R")

# read original data
outcomes1 = readRDS("data/summary/calculate-outcomes/calculate-outcomes-RNN.rds")
# read valid data
outcomes2 = readRDS("data/summary/break-SELFIES/calculate-outcomes-RNN-SELFIES-valid.rds") %>% 
  mutate(representation = fct_recode(constraints, 
                                     'Texas SELFIES' = 'c5', 
                                     'Unconstrained SELFIES' = 'none'))

# replace % valid
valid = readRDS("data/summary/break-SELFIES/SELFIES-percent-valid.rds") %>%
  mutate(outcome = '% valid') %>%
  mutate(representation = fct_recode(constraints,
                                     'Texas SELFIES' = 'c5',
                                     'Unconstrained SELFIES' = 'none'))
outcomes2 %<>% left_join(valid, by = c('database', 'enum_factor', 'n_molecules',
                                       'min_tc', 'sample_idx', 'outcome',
                                       'constraints', 'representation')) %>%
  mutate(value = ifelse(is.na(valid), value, valid))

# combine
outcomes = bind_rows(outcomes1, outcomes2) %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES', 
                                      'Texas SELFIES', 'Unconstrained SELFIES'))

# p-values
tests1 = outcomes %>% 
  filter(grepl('valid|stereoc|NP|Murck', outcome)) %>% 
  filter(database == 'ChEMBL',
         n_molecules == 1e5,
         enum_factor == 0,
         min_tc == 0) %>% 
  group_by(sample_idx, outcome) %>% 
  summarise(delta = value[representation == 'Texas SELFIES'] -
           value[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  group_by(outcome) %>% 
  do(tidy(t.test(.$delta)))
tests1

tests2 = outcomes %>% 
  filter(grepl('valid|stereoc|NP|Murck', outcome)) %>% 
  filter(database == 'ChEMBL',
         n_molecules == 1e5,
         enum_factor == 0,
         min_tc == 0) %>% 
  group_by(sample_idx, outcome) %>% 
  summarise(delta = value[representation == 'Unconstrained SELFIES'] -
           value[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  group_by(outcome) %>% 
  do(tidy(t.test(.$delta)))
tests2

###############################################################################-
## a-b. % valid ####
###############################################################################-

valid = filter(outcomes, outcome == '% valid')

representations = c('Texas SELFIES', 'Unconstrained SELFIES')
plots12 = map(representations, ~ {
  representation = .x
  exclude = fct_recode(representation,
                       'Unconstrained SELFIES' = 'Texas SELFIES', 
                       'Texas SELFIES' = 'Unconstrained SELFIES') %>%
    as.character()
  p = valid %>% 
    filter(database == 'ChEMBL',
           n_molecules == 1e5,
           enum_factor == 0,
           min_tc == 0) %>% 
    filter(representation != exclude) %>% 
    ggplot(aes(x = representation, y = value, fill = representation, 
               color = representation)) +
    geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
    geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
                stroke = 0.2) +
    scale_x_discrete(labels = ~ chartr(' ', '\n', .)) +
    scale_y_continuous('% valid', labels = ~ . * 100) +
    scale_color_manual(values = representation_pal) +
    scale_fill_manual(values = representation_pal) +
    boxed_theme() +
    theme(axis.title.x = element_blank(),
          axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = 'none',
          aspect.ratio = 1.4) +
    ggh4x::force_panelsizes(rows = unit(28.5, 'mm'),
                            cols = unit(28.5 / 1.4, 'mm'))
  p
})
p12 = wrap_plots(plots12, nrow = 1)
p12
ggsave("fig/supp-invalid-SELFIES/pct-valid-default.pdf", p12,
       width = 8, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## c-d. JSD Murcko ####
###############################################################################-

murcko = filter(outcomes, outcome == 'Jensen-Shannon distance, Murcko scaffolds')

representations = c('Texas SELFIES', 'Unconstrained SELFIES')
plots34 = map(representations, ~ {
  representation = .x
  exclude = fct_recode(representation,
                       'Unconstrained SELFIES' = 'Texas SELFIES', 
                       'Texas SELFIES' = 'Unconstrained SELFIES') %>%
    as.character()
  p = murcko %>% 
    filter(database == 'ChEMBL',
           n_molecules == 1e5,
           enum_factor == 0,
           min_tc == 0) %>% 
    filter(representation != exclude) %>% 
    ggplot(aes(x = representation, y = value, fill = representation, 
               color = representation)) +
    geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
    geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
                stroke = 0.2) +
    scale_x_discrete(labels = ~ chartr(' ', '\n', .)) +
    scale_y_continuous('JSD, Murcko scaffolds') +
    scale_color_manual(values = representation_pal) +
    scale_fill_manual(values = representation_pal) +
    boxed_theme() +
    theme(axis.title.x = element_blank(),
          axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = 'none',
          aspect.ratio = 1.4) +
    ggh4x::force_panelsizes(rows = unit(28.5, 'mm'),
                            cols = unit(28.5 / 1.4, 'mm'))
  p
})
p34 = wrap_plots(plots34, nrow = 1)
p34
ggsave("fig/supp-invalid-SELFIES/murcko-default.pdf", p34,
       width = 8, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## e-f. JSD NP-likeness ####
###############################################################################-

NP = filter(outcomes, outcome == 'Jensen-Shannon distance, NP score')

representations = c('Texas SELFIES', 'Unconstrained SELFIES')
plots56 = map(representations, ~ {
  representation = .x
  exclude = fct_recode(representation,
                       'Unconstrained SELFIES' = 'Texas SELFIES', 
                       'Texas SELFIES' = 'Unconstrained SELFIES') %>%
    as.character()
  p = NP %>% 
    filter(database == 'ChEMBL',
           n_molecules == 1e5,
           enum_factor == 0,
           min_tc == 0) %>% 
    filter(representation != exclude) %>% 
    ggplot(aes(x = representation, y = value, fill = representation, 
               color = representation)) +
    geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
    geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
                stroke = 0.2) +
    scale_x_discrete(labels = ~ chartr(' ', '\n', .)) +
    scale_y_continuous('JSD, natural product-likeness') +
    scale_color_manual(values = representation_pal) +
    scale_fill_manual(values = representation_pal) +
    boxed_theme() +
    theme(axis.title.x = element_blank(),
          axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = 'none',
          aspect.ratio = 1.4) +
    ggh4x::force_panelsizes(rows = unit(28.5, 'mm'),
                            cols = unit(28.5 / 1.4, 'mm'))
  p
})
p56 = wrap_plots(plots56, nrow = 1)
p56
ggsave("fig/supp-invalid-SELFIES/NP-default.pdf", p56,
       width = 8, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## g-h. JSD stereocentres ####
###############################################################################-

stereo = filter(outcomes, outcome == 'Jensen-Shannon distance, % stereocenters')

representations = c('Texas SELFIES', 'Unconstrained SELFIES')
plots78 = map(representations, ~ {
  representation = .x
  exclude = fct_recode(representation,
                       'Unconstrained SELFIES' = 'Texas SELFIES', 
                       'Texas SELFIES' = 'Unconstrained SELFIES') %>%
    as.character()
  p = stereo %>% 
    filter(database == 'ChEMBL',
           n_molecules == 1e5,
           enum_factor == 0,
           min_tc == 0) %>% 
    filter(representation != exclude) %>% 
    ggplot(aes(x = representation, y = value, fill = representation, 
               color = representation)) +
    geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
    geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
                stroke = 0.2) +
    scale_x_discrete(labels = ~ chartr(' ', '\n', .)) +
    scale_y_continuous('JSD, % stereocenters') +
    scale_color_manual(values = representation_pal) +
    scale_fill_manual(values = representation_pal) +
    boxed_theme() +
    theme(axis.title.x = element_blank(),
          axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = 'none',
          aspect.ratio = 1.4) +
    ggh4x::force_panelsizes(rows = unit(28.5, 'mm'),
                            cols = unit(28.5 / 1.4, 'mm'))
  p
})
p78 = wrap_plots(plots78, nrow = 1)
p78
ggsave("fig/supp-invalid-SELFIES/stereo-default.pdf", p78,
       width = 8, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## i. FCD, n_molecules ####
###############################################################################-

fcd = filter(outcomes, outcome == 'Frechet ChemNet distance',
             representation != 'Texas SELFIES') 
p9 = fcd %>% 
  filter(database == 'ChEMBL',
         n_molecules %in% c(3e4, 3e5),
         enum_factor == 0,
         min_tc == 0) %>% 
  mutate(n_molecules_str = ifelse(n_molecules == 3e4, '30,000 molecules',
                                  '300,000 molecules')) %>% 
  ggplot(aes(x = representation, y = value, fill = representation, 
             color = representation)) +
  facet_wrap(~ n_molecules_str, scales = 'free') +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_x_discrete(labels = ~ chartr(' ', '\n', .)) +
  scale_y_continuous('Frechet ChemNet distance') +
  scale_color_manual(values = representation_pal) +
  scale_fill_manual(values = representation_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1.4) +
  ggh4x::force_panelsizes(rows = unit(28.5, 'mm'),
                          cols = unit(28.5 / 1.4, 'mm'))
p9
ggsave("fig/supp-invalid-SELFIES/FCD-n_molecules-unconstrained.pdf", p9,
       width = 9, height = 4.6, units = "cm", useDingbats = FALSE)

fcd %>% 
  filter(database == 'ChEMBL',
         n_molecules %in% c(3e4, 3e5),
         enum_factor == 0,
         min_tc == 0) %>% 
  mutate(n_molecules_str = ifelse(n_molecules == 3e4, '30,000 molecules',
                                  '300,000 molecules')) %>% 
  group_by(n_molecules_str, sample_idx) %>% 
  summarise(delta = value[representation == 'Unconstrained SELFIES'] -
           value[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  group_by(n_molecules_str) %>% 
  do(tidy(t.test(.$delta)))
  
###############################################################################-
## j. FCD, database ####
###############################################################################-

p10 = fcd %>% 
  filter(database == 'GDB13',
         n_molecules == 1e5,
         enum_factor == 0,
         min_tc == 0) %>% 
  mutate(database = 'GDB-13') %>% 
  ggplot(aes(x = representation, y = value, fill = representation, 
             color = representation)) +
  facet_wrap(~ database, scales = 'free') +
  # geom_line(aes(group = factor(sample_idx))) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_x_discrete(labels = ~ chartr(' ', '\n', .)) +
  scale_y_continuous('Frechet ChemNet distance') +
  scale_color_manual(values = representation_pal) +
  scale_fill_manual(values = representation_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1.4) +
  ggh4x::force_panelsizes(rows = unit(28.5, 'mm'),
                          cols = unit(28.5 / 1.4, 'mm'))
p10
ggsave("fig/supp-invalid-SELFIES/FCD-database-unconstrained.pdf", p10,
       width = 9, height = 4.6, units = "cm", useDingbats = FALSE)

fcd %>% 
  filter(database == 'GDB13',
         n_molecules == 1e5,
         enum_factor == 0,
         min_tc == 0) %>% 
  mutate(database = 'GDB-13') %>% 
  group_by(database, sample_idx) %>% 
  summarise(delta = value[representation == 'Unconstrained SELFIES'] -
              value[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  group_by(database) %>% 
  do(tidy(t.test(.$delta)))

###############################################################################-
## k. FCD, min_Tc ####
###############################################################################-

p11 = fcd %>% 
  filter(database == 'ChEMBL',
         n_molecules == 1e5,
         enum_factor == 0,
         between(min_tc, 0.01, 0.16)) %>% 
  mutate(min_tc_str = paste0('Tc ≥ ', format(min_tc, digits = 2))) %>% 
  ggplot(aes(x = representation, y = value, fill = representation, 
             color = representation)) +
  facet_wrap(~ min_tc_str, scales = 'free') +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_x_discrete(labels = ~ chartr(' ', '\n', .)) +
  scale_y_continuous('Frechet ChemNet distance') +
  scale_color_manual(values = representation_pal) +
  scale_fill_manual(values = representation_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1.4) +
  ggh4x::force_panelsizes(rows = unit(28.5, 'mm'),
                          cols = unit(28.5 / 1.4, 'mm'))
p11
ggsave("fig/supp-invalid-SELFIES/FCD-min_Tc-unconstrained.pdf", p11,
       width = 9, height = 4.6, units = "cm", useDingbats = FALSE)

fcd %>% 
  filter(database == 'ChEMBL',
         n_molecules == 1e5,
         enum_factor == 0,
         between(min_tc, 0.01, 0.16)) %>% 
  mutate(min_tc_str = paste0('Tc ≥ ', format(min_tc, digits = 2))) %>% 
  group_by(min_tc_str, sample_idx) %>% 
  summarise(delta = value[representation == 'Unconstrained SELFIES'] -
              value[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  group_by(min_tc_str) %>% 
  do(tidy(t.test(.$delta)))

###############################################################################-
## l. FCD, enum_factor ####
###############################################################################-

p12 = fcd %>% 
  filter(database == 'ChEMBL',
         n_molecules == 1e5,
         enum_factor >= 10,
         min_tc == 0) %>% 
  mutate(enum_factor = paste0(enum_factor, 'x augmentation')) %>% 
  ggplot(aes(x = representation, y = value, fill = representation, 
             color = representation)) +
  facet_wrap(~ enum_factor, scales = 'free') +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_x_discrete(labels = ~ chartr(' ', '\n', .)) +
  scale_y_continuous('Frechet ChemNet distance') +
  scale_color_manual(values = representation_pal) +
  scale_fill_manual(values = representation_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1.4) +
  ggh4x::force_panelsizes(rows = unit(28.5, 'mm'),
                          cols = unit(28.5 / 1.4, 'mm'))
p12
ggsave("fig/supp-invalid-SELFIES/FCD-enum_factor-unconstrained.pdf", p12,
       width = 9, height = 4.6, units = "cm", useDingbats = FALSE)

fcd %>% 
  filter(database == 'ChEMBL',
         n_molecules == 1e5,
         enum_factor >= 10,
         min_tc == 0) %>% 
  mutate(enum_factor = paste0(enum_factor, 'x augmentation')) %>% 
  group_by(enum_factor, sample_idx) %>% 
  summarise(delta = value[representation == 'Unconstrained SELFIES'] -
              value[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  group_by(enum_factor) %>% 
  do(tidy(t.test(.$delta)))

###############################################################################-
## m. FCD, transformer ####
###############################################################################-

# read original data
outcomes3 = readRDS("data/summary/calculate-outcomes/calculate-outcomes-transformer.rds") 
# read valid data
outcomes4 = readRDS("data/summary/break-SELFIES/calculate-outcomes-transformer-SELFIES-valid.rds") %>% 
  mutate(representation = fct_recode(constraints, 
                                     'Texas SELFIES' = 'c5', 
                                     'Unconstrained SELFIES' = 'none'))
# combine
outcomes_t = bind_rows(outcomes3, outcomes4) %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES', 
                                      'Texas SELFIES', 'Unconstrained SELFIES'))

# plot
p13 = outcomes_t %>% 
  filter(outcome == 'Frechet ChemNet distance',
         representation != 'Texas SELFIES') %>% 
  mutate(model = 'Transformer') %>% 
  ggplot(aes(x = representation, y = value, fill = representation, 
             color = representation)) +
  facet_wrap(~ model, scales = 'free') +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) +
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black',
              stroke = 0.2) +
  scale_x_discrete(labels = ~ chartr(' ', '\n', .)) +
  scale_y_continuous('Frechet ChemNet distance') +
  scale_color_manual(values = representation_pal) +
  scale_fill_manual(values = representation_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1.4) +
  ggh4x::force_panelsizes(rows = unit(28.5, 'mm'),
                          cols = unit(28.5 / 1.4, 'mm'))
p13
ggsave("fig/supp-invalid-SELFIES/FCD-transformer-unconstrained.pdf", p13,
       width = 9, height = 4.6, units = "cm", useDingbats = FALSE)

outcomes_t %>% 
  filter(outcome == 'Frechet ChemNet distance') %>% 
  group_by(sample_idx) %>% 
  summarise(delta = value[representation == 'Unconstrained SELFIES'] -
              value[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  do(tidy(t.test(.$delta)))

###############################################################################-
## n. Cohen's d, n_molecules ####
###############################################################################-

# read data
eff = readRDS("data/summary/break-SELFIES/parse-valid-SELFIES-RNN.rds")$effects %>% 
  # filter to unconstrained
  filter(constraints == 'none')

# plot 
df14 = eff %>% 
  filter(analysis == 'loss') %>% 
  filter(database == 'ChEMBL',
         representation == 'SELFIES',
         n_molecules %in% c(3e4, 3e5),
         enum_factor == 0,
         min_tc == 0) %>% 
  mutate(n_molecules_str = ifelse(n_molecules == 3e4, '30,000\nmolecules',
                                  '300,000\nmolecules'))
p14 = df14 %>% 
  mutate(xval = 'Valid vs.\ninvalid\nSELFIES') %>% 
  ggplot(aes(x = xval, y = d, fill = representation, color = representation)) +
  facet_wrap(~ n_molecules_str, scales = 'free') +
  geom_hline(aes(yintercept = 0), color = 'grey88', size = 0.4) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous("Cohen's d") +
  scale_color_manual(values = '#bbbbbb') +
  scale_fill_manual(values = '#bbbbbb') +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        # axis.ticks.x = element_blank(),
        legend.position = 'none',
        aspect.ratio = 2 * 1.7) +
  ggh4x::force_panelsizes(rows = unit(28.5, 'mm'),
                          cols = unit(28.5 / (2 * 1.7), 'mm'))
p14
ggsave("fig/supp-invalid-SELFIES/loss-d-n_molecules.pdf", p14,
       width = 6, height = 6, units = "cm", useDingbats = FALSE)

# test
df14 %>% 
  group_by(n_molecules_str) %>% 
  do(tidy(t.test(.$d)))

###############################################################################-
## j. Cohen's d, database ####
###############################################################################-

# plot 
df15 = eff %>% 
  filter(analysis == 'loss') %>% 
  filter(database == 'GDB13',
         representation == 'SELFIES',
         n_molecules == 1e5,
         enum_factor == 0,
         min_tc == 0) %>% 
  mutate(database = 'GDB-13')
p15 = df15 %>% 
  mutate(xval = 'Valid vs.\ninvalid\nSELFIES') %>% 
  ggplot(aes(x = xval, y = d, fill = representation, color = representation)) +
  facet_wrap(~ database, scales = 'free') +
  geom_hline(aes(yintercept = 0), color = 'grey88', size = 0.4) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous("Cohen's d") +
  scale_color_manual(values = '#bbbbbb') +
  scale_fill_manual(values = '#bbbbbb') +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        # axis.ticks.x = element_blank(),
        legend.position = 'none',
        aspect.ratio = 2 * 1.7) +
  ggh4x::force_panelsizes(rows = unit(28.5, 'mm'),
                          cols = unit(28.5 / (2 * 1.7), 'mm'))
p15
ggsave("fig/supp-invalid-SELFIES/loss-d-database.pdf", p15,
       width = 4, height = 4.3, units = "cm", useDingbats = FALSE)

# test
df15 %>% 
  group_by(database) %>% 
  do(tidy(t.test(.$d)))

###############################################################################-
## k. Cohen's d, min_Tc ####
###############################################################################-

# plot 
df16 = eff %>% 
  filter(analysis == 'loss') %>% 
  filter(database == 'ChEMBL',
         representation == 'SELFIES',
         n_molecules == 1e5,
         enum_factor == 0,
         between(min_tc, 0.01, 0.16)) %>% 
  mutate(min_tc_str = paste0('Tc ≥ ', format(min_tc, digits = 2))) 
p16 = df16 %>% 
  mutate(xval = 'Valid vs.\ninvalid\nSELFIES') %>% 
  ggplot(aes(x = xval, y = d, fill = representation, color = representation)) +
  facet_wrap(~ min_tc_str, scales = 'free') +
  geom_hline(aes(yintercept = 0), color = 'grey88', size = 0.4) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous("Cohen's d") +
  scale_color_manual(values = '#bbbbbb') +
  scale_fill_manual(values = '#bbbbbb') +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        # axis.ticks.x = element_blank(),
        legend.position = 'none',
        aspect.ratio = 2 * 1.7) +
  ggh4x::force_panelsizes(rows = unit(28.5, 'mm'),
                          cols = unit(28.5 / (2 * 1.7), 'mm'))
p16
ggsave("fig/supp-invalid-SELFIES/loss-d-min_tc.pdf", p16,
       width = 6, height = 6, units = "cm", useDingbats = FALSE)

# test
df16 %>% 
  group_by(min_tc_str) %>% 
  do(tidy(t.test(.$d)))

###############################################################################-
## l. Cohen's d, enum_factor ####
###############################################################################-

# plot 
df17 = eff %>% 
  filter(analysis == 'loss') %>% 
  filter(database == 'ChEMBL',
         representation == 'SELFIES',
         n_molecules == 1e5,
         enum_factor >= 10,
         min_tc == 0) %>% 
  mutate(enum_factor = paste0(enum_factor, 'x\naugmentation')) 
p17 = df17 %>% 
  mutate(xval = 'Valid vs.\ninvalid\nSELFIES') %>% 
  ggplot(aes(x = xval, y = d, fill = representation, color = representation)) +
  facet_wrap(~ enum_factor, scales = 'free') +
  geom_hline(aes(yintercept = 0), color = 'grey88', size = 0.4) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous("Cohen's d") +
  scale_color_manual(values = '#bbbbbb') +
  scale_fill_manual(values = '#bbbbbb') +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        # axis.ticks.x = element_blank(),
        legend.position = 'none',
        aspect.ratio = 2 * 1.7) +
  ggh4x::force_panelsizes(rows = unit(28.5, 'mm'),
                          cols = unit(28.5 / (2 * 1.7), 'mm'))
p17
ggsave("fig/supp-invalid-SELFIES/loss-d-enum_factor.pdf", p16,
       width = 6, height = 6, units = "cm", useDingbats = FALSE)

# test
df17 %>% 
  group_by(enum_factor) %>% 
  do(tidy(t.test(.$d)))

###############################################################################-
## m. Cohen's d, MolGPT ####
###############################################################################-

# read data
eff2 = readRDS("data/summary/break-SELFIES/parse-valid-SELFIES-transformer.rds") %>% 
  # filter to unconstrained
  filter(constraints == 'none')

# plot 
df18 = eff2 %>% 
  filter(analysis == 'loss') %>% 
  mutate(model = 'Transformer')
stopifnot(nrow(df18) == 10)
p18 = df18 %>% 
  mutate(xval = 'Valid vs.\ninvalid\nSELFIES') %>% 
  ggplot(aes(x = xval, y = d, fill = representation, color = representation)) +
  facet_wrap(~ model, scales = 'free') +
  geom_hline(aes(yintercept = 0), color = 'grey88', size = 0.4) +
  geom_blank() + 
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) +
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black',
              stroke = 0.2) +
  scale_y_continuous("Cohen's d") +
  scale_color_manual(values = '#bbbbbb') +
  scale_fill_manual(values = '#bbbbbb') +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        # axis.ticks.x = element_blank(),
        legend.position = 'none',
        aspect.ratio = 2 * 1.7) +
  ggh4x::force_panelsizes(rows = unit(28.5, 'mm'),
                          cols = unit(28.5 / (2 * 1.7), 'mm'))
p18
ggsave("fig/supp-invalid-SELFIES/loss-d-transformer.pdf", p18,
       width = 4, height = 4.3, units = "cm", useDingbats = FALSE)

# test
df18 %>% 
  group_by(model) %>% 
  do(tidy(t.test(.$d)))
