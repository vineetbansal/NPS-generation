setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = F)
library(tidyverse)
library(magrittr)
library(broom)
source("R/theme.R")

# read data
dat = readRDS("data/summary/calculate-outcomes/calculate-outcomes-RNN.rds") 

###############################################################################-
## a-c. other metrics, default settings ####
## a. Murcko scaffolds, default settings ####
###############################################################################-

# extract Murcko
murcko = filter(dat, grepl('Murcko', outcome))

# plot default
df1 = murcko %>% 
  filter(database == 'ChEMBL',
         n_molecules == 1e5,
         enum_factor == 0,
         min_tc == 0) %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES'))
stopifnot(nrow(df1) == 20)
p1 = df1 %>% 
  ggplot(aes(x = representation, y = value, fill = representation, 
             color = representation)) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous('JSD, Murcko scaffolds') +
  scale_color_manual(values = representation_pal) +
  scale_fill_manual(values = representation_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1.7)
p1
ggsave("fig/supp-calculate-outcomes/murcko-default.pdf", p1,
       width = 4, height = 4, units = "cm", useDingbats = FALSE)

# t-test
test = df1 %>% 
  group_by(sample_idx) %>% 
  summarise(delta = value[representation == 'SMILES'] -
           value[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  do(tidy(t.test(.$delta)))
message('JSD Murcko, default settings: p = ', format(test$p.value, digits = 2))

###############################################################################-
## b. NP-likeness, default settings ####
###############################################################################-

# extract NP-likeness
np = filter(dat, grepl('NP', outcome))

# plot default
df2 = np %>% 
  filter(database == 'ChEMBL',
         n_molecules == 1e5,
         enum_factor == 0,
         min_tc == 0) %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES'))
stopifnot(nrow(df2) == 20)
p2 = df2 %>% 
  ggplot(aes(x = representation, y = value, fill = representation, 
             color = representation)) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous('JSD, natural product-likeness') +
  scale_color_manual(values = representation_pal) +
  scale_fill_manual(values = representation_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1.7)
p2
ggsave("fig/supp-calculate-outcomes/NP-default.pdf", p2,
       width = 4, height = 4, units = "cm", useDingbats = FALSE)

# t-test
test = df2 %>% 
  group_by(sample_idx) %>% 
  summarise(delta = value[representation == 'SMILES'] -
           value[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  do(tidy(t.test(.$delta)))
message('JSD NP-likeness, default settings: p = ', format(test$p.value, digits = 2))

###############################################################################-
## c. % stereocentres, default settings ####
###############################################################################-

# extract pct stereo
stereo = filter(dat, grepl('stereoc', outcome))

# plot default
df3 = stereo %>% 
  filter(database == 'ChEMBL',
         n_molecules == 1e5,
         enum_factor == 0,
         min_tc == 0) %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES'))
stopifnot(nrow(df3) == 20)
p3 = df3 %>% 
  ggplot(aes(x = representation, y = value, fill = representation, 
             color = representation)) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous('JSD, % stereocentres') +
  scale_color_manual(values = representation_pal) +
  scale_fill_manual(values = representation_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1.7)
p3
ggsave("fig/supp-calculate-outcomes/stereocentres-default.pdf", p3,
       width = 4, height = 4, units = "cm", useDingbats = FALSE)

# t-test
test = df3 %>% 
  group_by(sample_idx) %>% 
  summarise(delta = value[representation == 'SMILES'] -
           value[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  do(tidy(t.test(.$delta)))
message('JSD % stereocentres, default settings: p = ', 
        format(test$p.value, digits = 2))

###############################################################################-
## d. PC1, default settings ####
###############################################################################-

wide = dat %>% 
  filter(database == 'ChEMBL',
         n_molecules == 1e5,
         enum_factor == 0,
         min_tc == 0) %>% 
  filter(grepl('stereo|NP|valid|Frechet|Murcko', outcome)) %>% 
  spread(outcome, value)
library(CLMeval)
pc1 = pmap_dbl(wide, function(...) {
  current = tibble(...)
  calculate_PC1(pct_valid = current$`% valid`, 
                FCD = current$`Frechet ChemNet distance`,
                JSD_stereocenters = current$`Jensen-Shannon distance, % stereocenters`,
                JSD_murcko = current$`Jensen-Shannon distance, Murcko scaffolds`,
                JSD_NP = current$`Jensen-Shannon distance, NP score`)
})
wide %<>% mutate(pc1 = pc1,
                 representation = fct_relevel(representation, 'SMILES', 'SELFIES'))
p4 = wide %>% 
  ggplot(aes(x = representation, y = pc1, fill = representation, 
             color = representation)) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous('PC1') +
  scale_color_manual(values = representation_pal) +
  scale_fill_manual(values = representation_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1.7)
p4
ggsave("fig/supp-calculate-outcomes/PC1-default.pdf", p4,
       width = 4, height = 4, units = "cm", useDingbats = FALSE)

# t-test
test = wide %>% 
  group_by(sample_idx) %>% 
  summarise(delta = pc1[representation == 'SMILES'] -
           pc1[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  do(tidy(t.test(.$delta)))
message('PC1, default settings: p = ', 
        format(test$p.value, digits = 2))

###############################################################################-
## e-f. % valid, FCD, PC1 for n_molecules=5e4, 5e5 ####
## e. % valid for n_molecules=5e4, 5e5 ####
###############################################################################-

valid = filter(dat, grepl('valid', outcome))

# plot default
df5 = valid %>% 
  filter(database == 'ChEMBL',
         n_molecules %in% c(3e4, 3e5),
         enum_factor == 0,
         min_tc == 0) %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES'),
         n_molecules_str = ifelse(n_molecules == 3e4, '30,000 molecules',
                                  '300,000 molecules'))
stopifnot(nrow(df5) == 40)
p5 = df5 %>% 
  ggplot(aes(x = representation, y = value, fill = representation, 
             color = representation)) +
  facet_wrap(~ n_molecules_str, scales = 'free') +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous('% valid', labels = ~ . * 100,
                     breaks = pretty_breaks()) +
  scale_color_manual(values = representation_pal) +
  scale_fill_manual(values = representation_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1.7)
p5
ggsave("fig/supp-calculate-outcomes/pct-valid-n_molecules.pdf", p5,
       width = 8, height = 4.15, units = "cm", useDingbats = FALSE)

# t-test
tests = df5 %>% 
  group_by(sample_idx, n_molecules_str) %>% 
  filter(n_distinct(representation) == 2) %>% 
  summarise(delta = value[representation == 'SMILES'] -
           value[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  group_by(n_molecules_str) %>% 
  do(tidy(t.test(.$delta)))
pmap_dfr(tests, function(...) {
  current = tibble(...)
  message('% valid, n_molecules=', current$n_molecules_str, ': p = ', 
          format(current$p.value, digits = 2))
})

###############################################################################-
## f. FCD for n_molecules=5e4, 5e5 ####
###############################################################################-

fcd = filter(dat, grepl('ChemNet', outcome))

# plot default
df6 = fcd %>% 
  filter(database == 'ChEMBL',
         n_molecules %in% c(3e4, 3e5),
         enum_factor == 0,
         min_tc == 0) %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES'),
         n_molecules_str = ifelse(n_molecules == 3e4, '30,000 molecules',
                                  '300,000 molecules'))
stopifnot(nrow(df6) == 40)
p6 = df6 %>% 
  ggplot(aes(x = representation, y = value, fill = representation, 
             color = representation)) +
  facet_wrap(~ n_molecules_str, scales = 'free') +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous('Frechet ChemNet distance') +
  scale_color_manual(values = representation_pal) +
  scale_fill_manual(values = representation_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1.7)
p6
ggsave("fig/supp-calculate-outcomes/FCD-n_molecules.pdf", p6,
       width = 8, height = 4.15, units = "cm", useDingbats = FALSE)

# t-test
tests = df6 %>% 
  group_by(sample_idx, n_molecules_str) %>% 
  filter(n_distinct(representation) == 2) %>% 
  summarise(delta = value[representation == 'SMILES'] -
           value[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  group_by(n_molecules_str) %>% 
  do(tidy(t.test(.$delta)))
pmap_dfr(tests, function(...) {
  current = tibble(...)
  message('FCD, n_molecules=', current$n_molecules_str, ': p = ', 
          format(current$p.value, digits = 2))
})

###############################################################################-
## g. PC1 for n_molecules=5e4, 5e5 ####
###############################################################################-

df7 = dat %>% 
  filter(database == 'ChEMBL',
         n_molecules %in% c(3e4, 3e5),
         enum_factor == 0,
         min_tc == 0) %>% 
  filter(grepl('stereo|NP|valid|Frechet|Murcko', outcome)) %>% 
  spread(outcome, value)
library(CLMeval)
pc1 = pmap_dbl(df7, function(...) {
  current = tibble(...)
  calculate_PC1(pct_valid = current$`% valid`, 
                FCD = current$`Frechet ChemNet distance`,
                JSD_stereocenters = current$`Jensen-Shannon distance, % stereocenters`,
                JSD_murcko = current$`Jensen-Shannon distance, Murcko scaffolds`,
                JSD_NP = current$`Jensen-Shannon distance, NP score`)
})
df7 %<>%
  mutate(pc1 = pc1,
         representation = fct_relevel(representation, 'SMILES', 'SELFIES'),
         n_molecules_str = ifelse(n_molecules == 3e4, '30,000 molecules',
                                  '300,000 molecules'))
p7 = df7 %>% 
  ggplot(aes(x = representation, y = pc1, fill = representation, 
             color = representation)) +
  facet_wrap(~ n_molecules_str, scales = 'free') +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous('PC1') +
  scale_color_manual(values = representation_pal) +
  scale_fill_manual(values = representation_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1.7)
p7
ggsave("fig/supp-calculate-outcomes/PC1-n_molecules.pdf", p7,
       width = 8, height = 4.15, units = "cm", useDingbats = FALSE)

# t-test
tests = df7 %>% 
  group_by(sample_idx, n_molecules_str) %>% 
  filter(n_distinct(representation) == 2) %>% 
  summarise(delta = pc1[representation == 'SMILES'] -
           pc1[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  group_by(n_molecules_str) %>% 
  do(tidy(t.test(.$delta)))
pmap_dfr(tests, function(...) {
  current = tibble(...)
  message('PC1, n_molecules=', current$n_molecules_str, ': p = ', 
          format(current$p.value, digits = 2))
})

###############################################################################-
## h-j. % valid, FCD, PC1 for database=GDB-13 ####
## h. % valid for database=GDB-13 ####
###############################################################################-

# plot default
df8 = valid %>% 
  filter(database == 'GDB13',
         n_molecules == 1e5,
         enum_factor == 0,
         min_tc == 0) %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES'),
         database = 'GDB-13')
stopifnot(nrow(df8) == 20)
p8 = df8 %>% 
  ggplot(aes(x = representation, y = value, fill = representation, 
             color = representation)) +
  facet_wrap(~ database) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous('% valid', labels = ~ . * 100,
                     breaks = pretty_breaks()) +
  scale_color_manual(values = representation_pal) +
  scale_fill_manual(values = representation_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1.7)
p8
ggsave("fig/supp-calculate-outcomes/pct-valid-database.pdf", p8,
       width = 8, height = 4.15, units = "cm", useDingbats = FALSE)

# t-test
tests = df8 %>% 
  group_by(sample_idx, database) %>% 
  filter(n_distinct(representation) == 2) %>% 
  summarise(delta = value[representation == 'SMILES'] -
           value[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  group_by(database) %>% 
  do(tidy(t.test(.$delta)))
pmap_dfr(tests, function(...) {
  current = tibble(...)
  message('% valid, database=', current$database, ': p = ', 
          format(current$p.value, digits = 2))
})

###############################################################################-
## i. FCD for database=GDB-13 ####
###############################################################################-

# plot default
df9 = fcd %>% 
  filter(database == 'GDB13',
         n_molecules == 1e5,
         enum_factor == 0,
         min_tc == 0) %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES'),
         database = 'GDB-13')
stopifnot(nrow(df9) == 20)
p9 = df9 %>% 
  ggplot(aes(x = representation, y = value, fill = representation, 
             color = representation)) +
  facet_wrap(~ database) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous('Frechet ChemNet distance') +
  scale_color_manual(values = representation_pal) +
  scale_fill_manual(values = representation_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1.7)
p9
ggsave("fig/supp-calculate-outcomes/FCD-database.pdf", p9,
       width = 8, height = 4.15, units = "cm", useDingbats = FALSE)

# t-test
tests = df9 %>% 
  group_by(sample_idx, database) %>% 
  filter(n_distinct(representation) == 2) %>% 
  summarise(delta = value[representation == 'SMILES'] -
           value[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  group_by(database) %>% 
  do(tidy(t.test(.$delta)))
pmap_dfr(tests, function(...) {
  current = tibble(...)
  message('FCD, database=', current$database, ': p = ', 
          format(current$p.value, digits = 2))
})

###############################################################################-
## j. PC1 for database=GDB13 ####
###############################################################################-

df10 = dat %>% 
  filter(database == 'GDB13',
         n_molecules == 1e5,
         enum_factor == 0,
         min_tc == 0) %>% 
  filter(grepl('stereo|NP|valid|Frechet|Murcko', outcome)) %>% 
  spread(outcome, value)
library(CLMeval)
pc1 = pmap_dbl(df10, function(...) {
  current = tibble(...)
  calculate_PC1(pct_valid = current$`% valid`, 
                FCD = current$`Frechet ChemNet distance`,
                JSD_stereocenters = current$`Jensen-Shannon distance, % stereocenters`,
                JSD_murcko = current$`Jensen-Shannon distance, Murcko scaffolds`,
                JSD_NP = current$`Jensen-Shannon distance, NP score`)
})
df10 %<>%
  mutate(pc1 = pc1,
         representation = fct_relevel(representation, 'SMILES', 'SELFIES'),
         database = 'GDB-13')
p10 = df10 %>% 
  ggplot(aes(x = representation, y = pc1, fill = representation, 
             color = representation)) +
  facet_wrap(~ database, scales = 'free') +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous('PC1') +
  scale_color_manual(values = representation_pal) +
  scale_fill_manual(values = representation_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1.7)
p10
ggsave("fig/supp-calculate-outcomes/PC1-database.pdf", p10,
       width = 8, height = 4.15, units = "cm", useDingbats = FALSE)

# t-test
tests = df10 %>% 
  group_by(sample_idx, database) %>% 
  filter(n_distinct(representation) == 2) %>% 
  summarise(delta = pc1[representation == 'SMILES'] -
           pc1[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  group_by(database) %>% 
  do(tidy(t.test(.$delta)))
pmap_dfr(tests, function(...) {
  current = tibble(...)
  message('PC1, database=', current$database, ': p = ', 
          format(current$p.value, digits = 2))
})

###############################################################################-
## k-m. % valid, FCD, PC1 for min_tc=0.05, 0.1, 0.15 ####
## k. % valid for min_tc=0.05, 0.1, 0.15 ####
###############################################################################-

# plot default
df11 = valid %>% 
  filter(database == 'ChEMBL',
         n_molecules == 1e5,
         enum_factor == 0,
         between(min_tc, 0.01, 0.16)) %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES'),
         min_tc_str = paste0('Tc ≥ ', format(min_tc, digits = 2)))
stopifnot(nrow(df11) == 60)
p11 = df11 %>% 
  ggplot(aes(x = representation, y = value, fill = representation, 
             color = representation)) +
  facet_wrap(~ min_tc_str, scales = 'free') +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous('% valid', labels = ~ . * 100,
                     breaks = pretty_breaks()) +
  scale_color_manual(values = representation_pal) +
  scale_fill_manual(values = representation_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1.7)
p11
ggsave("fig/supp-calculate-outcomes/pct-valid-min_Tc.pdf", p11,
       width = 8, height = 4.15, units = "cm", useDingbats = FALSE)

# t-test
tests = df11 %>% 
  group_by(sample_idx, min_tc_str) %>% 
  filter(n_distinct(representation) == 2) %>% 
  summarise(delta = value[representation == 'SMILES'] -
           value[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  group_by(min_tc_str) %>% 
  do(tidy(t.test(.$delta)))
pmap_dfr(tests, function(...) {
  current = tibble(...)
  message('% valid, min_tc_str=', current$min_tc_str, ': p = ', 
          format(current$p.value, digits = 2))
})

###############################################################################-
## l. FCD for min_Tc=0.05,0.1,0.15 ####
###############################################################################-

# plot default
df12 = fcd %>% 
  filter(database == 'ChEMBL',
         n_molecules == 1e5,
         enum_factor == 0,
         between(min_tc, 0.01, 0.16)) %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES'),
         min_tc_str = paste0('Tc ≥ ', format(min_tc, digits = 2)))
stopifnot(nrow(df12) == 60)
p12 = df12 %>% 
  ggplot(aes(x = representation, y = value, fill = representation, 
             color = representation)) +
  facet_wrap(~ min_tc_str, scales = 'free') +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous('Frechet ChemNet distance') +
  scale_color_manual(values = representation_pal) +
  scale_fill_manual(values = representation_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1.7)
p12
ggsave("fig/supp-calculate-outcomes/FCD-min_Tc.pdf", p12,
       width = 8, height = 4.15, units = "cm", useDingbats = FALSE)

# t-test
tests = df12 %>% 
  group_by(sample_idx, min_tc_str) %>% 
  filter(n_distinct(representation) == 2) %>% 
  summarise(delta = value[representation == 'SMILES'] -
           value[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  group_by(min_tc_str) %>% 
  do(tidy(t.test(.$delta)))
pmap_dfr(tests, function(...) {
  current = tibble(...)
  message('FCD, min_tc_str=', current$min_tc_str, ': p = ', 
          format(current$p.value, digits = 2))
})

###############################################################################-
## m. PC1 for min_Tc=0.05,0.1,0.15 ####
###############################################################################-

df13 = dat %>% 
  filter(database == 'ChEMBL',
         n_molecules == 1e5,
         enum_factor == 0,
         between(min_tc, 0.01, 0.16)) %>% 
  filter(grepl('stereo|NP|valid|Frechet|Murcko', outcome)) %>% 
  spread(outcome, value)
library(CLMeval)
pc1 = pmap_dbl(df13, function(...) {
  current = tibble(...)
  calculate_PC1(pct_valid = current$`% valid`, 
                FCD = current$`Frechet ChemNet distance`,
                JSD_stereocenters = current$`Jensen-Shannon distance, % stereocenters`,
                JSD_murcko = current$`Jensen-Shannon distance, Murcko scaffolds`,
                JSD_NP = current$`Jensen-Shannon distance, NP score`)
})
df13 %<>%
  mutate(pc1 = pc1,
         representation = fct_relevel(representation, 'SMILES', 'SELFIES'),
         min_tc_str = paste0('Tc ≥ ', format(min_tc, digits = 2)))
p13 = df13 %>% 
  ggplot(aes(x = representation, y = pc1, fill = representation, 
             color = representation)) +
  facet_wrap(~ min_tc_str, scales = 'free') +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous('PC1') +
  scale_color_manual(values = representation_pal) +
  scale_fill_manual(values = representation_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1.7)
p13
ggsave("fig/supp-calculate-outcomes/PC1-min_Tc.pdf", p13,
       width = 8, height = 4.15, units = "cm", useDingbats = FALSE)

# t-test
tests = df13 %>% 
  group_by(sample_idx, min_tc_str) %>% 
  filter(n_distinct(representation) == 2) %>% 
  summarise(delta = pc1[representation == 'SMILES'] -
           pc1[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  group_by(min_tc_str) %>% 
  do(tidy(t.test(.$delta)))
pmap_dfr(tests, function(...) {
  current = tibble(...)
  message('PC1, min_tc_str=', current$min_tc_str, ': p = ', 
          format(current$p.value, digits = 2))
})

###############################################################################-
## n-p. % valid, FCD, PC1 for enum_factor=10, 30 ####
## n. % valid for enum_factor=10, 30 ####
###############################################################################-

# plot default
df14 = valid %>% 
  filter(database == 'ChEMBL',
         n_molecules == 1e5,
         enum_factor >= 10,
         min_tc == 0) %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES'),
         enum_factor = paste0(enum_factor, 'x augmentation'))
stopifnot(nrow(df14) == 40)
p14 = df14 %>% 
  ggplot(aes(x = representation, y = value, fill = representation, 
             color = representation)) +
  facet_wrap(~ enum_factor, scales = 'free') +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous('% valid', labels = ~ . * 100,
                     breaks = pretty_breaks()) +
  scale_color_manual(values = representation_pal) +
  scale_fill_manual(values = representation_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1.7)
p14
ggsave("fig/supp-calculate-outcomes/pct-valid-enum_factor.pdf", p14,
       width = 8, height = 4.15, units = "cm", useDingbats = FALSE)

# t-test
tests = df14 %>% 
  group_by(sample_idx, enum_factor) %>% 
  filter(n_distinct(representation) == 2) %>% 
  summarise(delta = value[representation == 'SMILES'] -
           value[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  group_by(enum_factor) %>% 
  do(tidy(t.test(.$delta)))
pmap_dfr(tests, function(...) {
  current = tibble(...)
  message('% valid, enum_factor=', current$enum_factor, ': p = ', 
          format(current$p.value, digits = 2))
})

###############################################################################-
## o. FCD for enum_factor=10, 30 ####
###############################################################################-

# plot default
df15 = fcd %>% 
  filter(database == 'ChEMBL',
         n_molecules == 1e5,
         enum_factor >= 10,
         min_tc == 0) %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES'),
         enum_factor = paste0(enum_factor, 'x augmentation'))
stopifnot(nrow(df15) == 40)
p15 = df15 %>% 
  ggplot(aes(x = representation, y = value, fill = representation, 
             color = representation)) +
  facet_wrap(~ enum_factor, scales = 'free') +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous('Frechet ChemNet distance') +
  scale_color_manual(values = representation_pal) +
  scale_fill_manual(values = representation_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1.7)
p15
ggsave("fig/supp-calculate-outcomes/FCD-enum_factor.pdf", p15,
       width = 8, height = 4.15, units = "cm", useDingbats = FALSE)

# t-test
tests = df15 %>% 
  group_by(sample_idx, enum_factor) %>% 
  filter(n_distinct(representation) == 2) %>% 
  summarise(delta = value[representation == 'SMILES'] -
           value[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  group_by(enum_factor) %>% 
  do(tidy(t.test(.$delta)))
pmap_dfr(tests, function(...) {
  current = tibble(...)
  message('FCD, enum_factor=', current$enum_factor, ': p = ', 
          format(current$p.value, digits = 2))
})

###############################################################################-
## p. PC1 for enum_factor=10,30 ####
###############################################################################-

df16 = dat %>% 
  filter(database == 'ChEMBL',
         n_molecules == 1e5,
         enum_factor >= 10,
         min_tc == 0) %>% 
  filter(grepl('stereo|NP|valid|Frechet|Murcko', outcome)) %>% 
  spread(outcome, value)
library(CLMeval)
pc1 = pmap_dbl(df16, function(...) {
  current = tibble(...)
  calculate_PC1(pct_valid = current$`% valid`, 
                FCD = current$`Frechet ChemNet distance`,
                JSD_stereocenters = current$`Jensen-Shannon distance, % stereocenters`,
                JSD_murcko = current$`Jensen-Shannon distance, Murcko scaffolds`,
                JSD_NP = current$`Jensen-Shannon distance, NP score`)
})
df16 %<>%
  mutate(pc1 = pc1,
         representation = fct_relevel(representation, 'SMILES', 'SELFIES'),
         enum_factor = paste0(enum_factor, 'x augmentation'))
p16 = df16 %>% 
  ggplot(aes(x = representation, y = pc1, fill = representation, 
             color = representation)) +
  facet_wrap(~ enum_factor, scales = 'free') +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous('PC1') +
  scale_color_manual(values = representation_pal) +
  scale_fill_manual(values = representation_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1.7)
p16
ggsave("fig/supp-calculate-outcomes/PC1-enum_factor.pdf", p16,
       width = 8, height = 4.15, units = "cm", useDingbats = FALSE)

# t-test
tests = df16 %>% 
  group_by(sample_idx, enum_factor) %>% 
  filter(n_distinct(representation) == 2) %>% 
  summarise(delta = pc1[representation == 'SMILES'] -
           pc1[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  group_by(enum_factor) %>% 
  do(tidy(t.test(.$delta)))
pmap_dfr(tests, function(...) {
  current = tibble(...)
  message('PC1, enum_factor=', current$enum_factor, ': p = ', 
          format(current$p.value, digits = 2))
})


###############################################################################-
## q-s. % valid, FCD, PC1 for MolGPT ####
## q. % valid for model=MolGPT ####
###############################################################################-

# read data
dat2 = readRDS("data/summary/calculate-outcomes/calculate-outcomes-transformer.rds") 
valid2 = filter(dat2, outcome == '% valid')

# plot default
df17 = valid2 %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES'),
         model = 'Transformer')
stopifnot(nrow(df17) == 20)
p17 = df17 %>% 
  ggplot(aes(x = representation, y = value, fill = representation, 
             color = representation)) +
  facet_wrap(~ model) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) +
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black',
              stroke = 0.2) +
  scale_y_continuous('% valid', labels = ~ . * 100,
                     breaks = pretty_breaks()) +
  scale_color_manual(values = representation_pal) +
  scale_fill_manual(values = representation_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1.7)
p17
ggsave("fig/supp-calculate-outcomes/pct-valid-transformer.pdf", p17,
       width = 8, height = 4.15, units = "cm", useDingbats = FALSE)

# t-test
tests = df17 %>%
  group_by(sample_idx, model) %>%
  filter(n_distinct(representation) == 2) %>%
  summarise(delta = value[representation == 'SMILES'] -
              value[representation == 'SELFIES']) %>%
  ungroup() %>%
  group_by(model) %>%
  do(tidy(t.test(.$delta)))
pmap_dfr(tests, function(...) {
  current = tibble(...)
  message('% valid, model=', current$model, ': p = ',
          format(current$p.value, digits = 2))
})

###############################################################################-
## i. FCD for model=MolGPT ####
###############################################################################-

fcd2 = filter(dat2, grepl('ChemNet', outcome))

# plot default
df18 = fcd2 %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES'),
         model = 'Transformer')
stopifnot(nrow(df18) == 20)
p18 = df18 %>% 
  ggplot(aes(x = representation, y = value, fill = representation, 
             color = representation)) +
  facet_wrap(~ model) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous('Frechet ChemNet distance') +
  scale_color_manual(values = representation_pal) +
  scale_fill_manual(values = representation_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1.7)
p18
ggsave("fig/supp-calculate-outcomes/FCD-transformer.pdf", p18,
       width = 8, height = 4.15, units = "cm", useDingbats = FALSE)

# t-test
tests = df18 %>% 
  group_by(sample_idx, model) %>% 
  filter(n_distinct(representation) == 2) %>% 
  summarise(delta = value[representation == 'SMILES'] -
              value[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  group_by(model) %>% 
  do(tidy(t.test(.$delta)))
pmap_dfr(tests, function(...) {
  current = tibble(...)
  message('FCD, model=', current$model, ': p = ', 
          format(current$p.value, digits = 2))
})

###############################################################################-
## j. PC1 for model=GDB13 ####
###############################################################################-

df19 = dat2 %>% 
  filter(grepl('stereo|NP|valid|Frechet|Murcko', outcome)) %>% 
  spread(outcome, value)
library(CLMeval)
pc1 = pmap_dbl(df19, function(...) {
  current = tibble(...)
  calculate_PC1(pct_valid = current$`% valid`, 
                FCD = current$`Frechet ChemNet distance`,
                JSD_stereocenters = current$`Jensen-Shannon distance, % stereocenters`,
                JSD_murcko = current$`Jensen-Shannon distance, Murcko scaffolds`,
                JSD_NP = current$`Jensen-Shannon distance, NP score`)
})
df19 %<>%
  mutate(pc1 = pc1,
         representation = fct_relevel(representation, 'SMILES', 'SELFIES'),
         model = 'Transformer')
stopifnot(nrow(df19) == 20)
p19 = df19 %>% 
  ggplot(aes(x = representation, y = pc1, fill = representation, 
             color = representation)) +
  facet_wrap(~ model, scales = 'free') +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous('PC1') +
  scale_color_manual(values = representation_pal) +
  scale_fill_manual(values = representation_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1.7)
p19
ggsave("fig/supp-calculate-outcomes/PC1-transformer.pdf", p19,
       width = 8, height = 4.15, units = "cm", useDingbats = FALSE)

# t-test
tests = df19 %>% 
  group_by(sample_idx, model) %>% 
  filter(n_distinct(representation) == 2) %>% 
  summarise(delta = pc1[representation == 'SMILES'] -
              pc1[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  group_by(model) %>% 
  do(tidy(t.test(.$delta)))
pmap_dfr(tests, function(...) {
  current = tibble(...)
  message('PC1, model=', current$model, ': p = ', 
          format(current$p.value, digits = 2))
})
