setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = F)
library(tidyverse)
library(magrittr)
library(broom)
source("R/theme.R")

# read effect sizes
eff = readRDS("data/summary/sample-molecules/parse-valid-SMILES-RNN.rds")$effects

###############################################################################-
## a-b. n_molecules=5e4, 5e5 ####
###############################################################################-

# plot n_molecules = 3e5, 3e6
df1 = eff %>% 
  filter(analysis == 'loss') %>% 
  filter(database == 'ChEMBL',
         representation == 'SMILES',
         n_molecules %in% c(3e4, 3e5),
         enum_factor == 0,
         min_tc == 0) %>% 
  mutate(n_molecules_str = ifelse(n_molecules == 3e4, '30,000\nmolecules',
                                  '300,000\nmolecules'))
p1 = df1 %>% 
  mutate(xval = 'Valid vs.\ninvalid\nSMILES') %>% 
  ggplot(aes(x = xval, y = d, fill = representation, 
             color = representation)) +
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
        legend.position = 'none',
        aspect.ratio = 2 * 1.7)
p1
ggsave("fig/supp-invalid-SMILES-losses/loss-d-n_molecules.pdf", p1,
       width = 4, height = 4, units = "cm", useDingbats = FALSE)

df2 = eff %>% 
  filter(grepl("loss: ", analysis)) %>% 
  mutate(xval = gsub("^.*: ", "", analysis) %>% str_to_sentence()) %>% 
  filter(database == 'ChEMBL',
         representation == 'SMILES',
         n_molecules %in% c(3e4, 3e5),
         enum_factor == 0,
         min_tc == 0) %>% 
  mutate(n_molecules_str = ifelse(n_molecules == 3e4, '30,000\nmolecules',
                                  '300,000\nmolecules'))
p2 = df2 %>% 
  ggplot(aes(x = xval, y = d, fill = xval, color = xval)) +
  facet_wrap(~ n_molecules_str, scales = 'free') +
  geom_hline(aes(yintercept = 0), color = 'grey88', size = 0.4) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous("Cohen's d") +
  scale_color_manual(values = error_type_pal) +
  scale_fill_manual(values = error_type_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1)
p2
ggsave("fig/supp-invalid-SMILES-losses/loss-type-d-n_molecules.pdf", p2,
       width = 6, height = 4, units = "cm", useDingbats = FALSE)

# tests
df1 %>% 
  group_by(n_molecules) %>% 
  do(tidy(t.test(.$d)))
df2 %>% 
  group_by(n_molecules, xval) %>% 
  do(tidy(t.test(.$d))) %>% 
  ungroup() %>% 
  arrange(p.value)

###############################################################################-
## c-d. database=GDB13 ####
###############################################################################-

# plot database = GDB-13
df3 = eff %>% 
  filter(analysis == 'loss') %>% 
  filter(database == 'GDB13',
         representation == 'SMILES',
         n_molecules == 1e5,
         enum_factor == 0,
         min_tc == 0) %>% 
  mutate(database = 'GDB-13')
p3 = df3 %>% 
  mutate(xval = 'Valid vs.\ninvalid\nSMILES') %>% 
  ggplot(aes(x = xval, y = d, fill = representation, 
             color = representation)) +
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
        legend.position = 'none',
        aspect.ratio = 2 * 1.7)
p3
ggsave("fig/supp-invalid-SMILES-losses/loss-d-database.pdf", p3,
       width = 4, height = 4, units = "cm", useDingbats = FALSE)

df4 = eff %>% 
  filter(grepl("loss: ", analysis)) %>% 
  mutate(xval = gsub("^.*: ", "", analysis) %>% str_to_sentence()) %>% 
  filter(database == 'GDB13',
         representation == 'SMILES',
         n_molecules == 1e5,
         enum_factor == 0,
         min_tc == 0) %>% 
  mutate(database = 'GDB-13')
p4 = df4 %>% 
  ggplot(aes(x = xval, y = d, fill = xval, color = xval)) +
  facet_wrap(~ database, scales = 'free') +
  geom_hline(aes(yintercept = 0), color = 'grey88', size = 0.4) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous("Cohen's d") +
  scale_color_manual(values = error_type_pal) +
  scale_fill_manual(values = error_type_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1)
p4
ggsave("fig/supp-invalid-SMILES-losses/loss-type-d-database.pdf", p4,
       width = 6, height = 4, units = "cm", useDingbats = FALSE)

# tests
df3 %>% 
  group_by(n_molecules) %>% 
  do(tidy(t.test(.$d)))
df4 %>% 
  group_by(n_molecules, xval) %>% 
  do(tidy(t.test(.$d))) %>% 
  ungroup() %>% 
  arrange(p.value)

###############################################################################-
## e-f. min_Tc=0.05,0.1,0.15 ####
###############################################################################-

# plot min_Tc
df5 = eff %>% 
  filter(analysis == 'loss') %>% 
  filter(database == 'ChEMBL',
         representation == 'SMILES',
         n_molecules == 1e5,
         enum_factor == 0,
         between(min_tc, 0.01, 0.16)) %>% 
  mutate(min_tc_str = paste0('Tc ≥ ', format(min_tc, digits = 2)))
p5 = df5 %>% 
  mutate(xval = 'Valid vs.\ninvalid\nSMILES') %>% 
  ggplot(aes(x = xval, y = d, fill = representation, 
             color = representation)) +
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
        legend.position = 'none',
        aspect.ratio = 2 * 1.7)
p5
ggsave("fig/supp-invalid-SMILES-losses/loss-d-min_Tc.pdf", p5,
       width = 6, height = 4, units = "cm", useDingbats = FALSE)

df6 = eff %>% 
  filter(grepl("loss: ", analysis)) %>% 
  mutate(xval = gsub("^.*: ", "", analysis) %>% str_to_sentence()) %>% 
  filter(database == 'ChEMBL',
         representation == 'SMILES',
         n_molecules == 1e5,
         enum_factor == 0,
         between(min_tc, 0.01, 0.16)) %>% 
  mutate(min_tc_str = paste0('Tc ≥ ', format(min_tc, digits = 2)))
p6 = df6 %>% 
  ggplot(aes(x = xval, y = d, fill = xval, color = xval)) +
  facet_wrap(~ min_tc_str, scales = 'free') +
  geom_hline(aes(yintercept = 0), color = 'grey88', size = 0.4) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous("Cohen's d") +
  scale_color_manual(values = error_type_pal) +
  scale_fill_manual(values = error_type_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1)
p6
ggsave("fig/supp-invalid-SMILES-losses/loss-type-d-min_Tc.pdf", p6,
       width = 9, height = 4, units = "cm", useDingbats = FALSE)

# tests
df5 %>% 
  group_by(min_tc_str) %>% 
  do(tidy(t.test(.$d))) %>% 
  arrange(p.value)
df6 %>% 
  group_by(min_tc_str, xval) %>% 
  do(tidy(t.test(.$d))) %>% 
  ungroup() %>% 
  arrange(p.value)

###############################################################################-
## g-h. enum_factor=10,30 ####
###############################################################################-

# plot enum_factor
df7 = eff %>% 
  filter(analysis == 'loss') %>% 
  filter(database == 'ChEMBL',
         representation == 'SMILES',
         n_molecules == 1e5,
         enum_factor >= 10,
         min_tc == 0) %>% 
  mutate(enum_factor = paste0(enum_factor, 'x\naugmentation'))
p7 = df7 %>% 
  mutate(xval = 'Valid vs.\ninvalid\nSMILES') %>% 
  ggplot(aes(x = xval, y = d, fill = representation, 
             color = representation)) +
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
        legend.position = 'none',
        aspect.ratio = 2 * 1.7)
p7
ggsave("fig/supp-invalid-SMILES-losses/loss-d-enum_factor.pdf", p7,
       width = 6, height = 4, units = "cm", useDingbats = FALSE)

df8 = eff %>% 
  filter(grepl("loss: ", analysis)) %>% 
  mutate(xval = gsub("^.*: ", "", analysis) %>% str_to_sentence()) %>% 
  filter(database == 'ChEMBL',
         representation == 'SMILES',
         n_molecules == 1e5,
         enum_factor >= 10,
         min_tc == 0) %>% 
  mutate(enum_factor = paste0(enum_factor, 'x\naugmentation'))
p8 = df8 %>% 
  ggplot(aes(x = xval, y = d, fill = xval, color = xval)) +
  facet_wrap(~ enum_factor, scales = 'free') +
  geom_hline(aes(yintercept = 0), color = 'grey88', size = 0.4) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous("Cohen's d") +
  scale_color_manual(values = error_type_pal) +
  scale_fill_manual(values = error_type_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1)
p8
ggsave("fig/supp-invalid-SMILES-losses/loss-type-d-enum_factor.pdf", p8,
       width = 9, height = 4, units = "cm", useDingbats = FALSE)


# tests
df7 %>% 
  group_by(enum_factor) %>% 
  do(tidy(t.test(.$d))) %>% 
  arrange(p.value)
df8 %>% 
  group_by(enum_factor, xval) %>% 
  do(tidy(t.test(.$d))) %>% 
  ungroup() %>% 
  arrange(p.value)

###############################################################################-
## i-j. MolGPT ####
###############################################################################-

# read effect sizes
eff2 = readRDS("data/summary/sample-molecules/parse-valid-SMILES-transformer.rds")

df9 = eff2 %>% 
  filter(analysis == 'loss') %>% 
  mutate(model = 'Transformer')
stopifnot(nrow(df9) == 10)
p9 = df9 %>% 
  mutate(xval = 'Valid vs.\ninvalid\nSMILES') %>% 
  ggplot(aes(x = xval, y = d, fill = representation, 
             color = representation)) +
  facet_wrap(~ model, scales = 'free') +
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
        legend.position = 'none',
        aspect.ratio = 2 * 1.7)
p9
ggsave("fig/supp-invalid-SMILES-losses/loss-d-transformer.pdf", p9,
       width = 6, height = 4, units = "cm", useDingbats = FALSE)

df10 = eff2 %>% 
  filter(grepl("loss: ", analysis)) %>% 
  mutate(xval = gsub("^.*: ", "", analysis) %>% str_to_sentence()) %>% 
  mutate(model = 'Transformer')
stopifnot(nrow(df10) == 60)
p10 = df10 %>% 
  ggplot(aes(x = xval, y = d, fill = xval, color = xval)) +
  facet_wrap(~ model, scales = 'free') +
  geom_hline(aes(yintercept = 0), color = 'grey88', size = 0.4) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous("Cohen's d") +
  scale_color_manual(values = error_type_pal) +
  scale_fill_manual(values = error_type_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1)
p10
ggsave("fig/supp-invalid-SMILES-losses/loss-type-d-transformer.pdf", p8,
       width = 9, height = 4, units = "cm", useDingbats = FALSE)

# tests
df9 %>% 
  group_by(model) %>% 
  do(tidy(t.test(.$d))) %>% 
  arrange(p.value)
df10 %>% 
  group_by(model, xval) %>% 
  do(tidy(t.test(.$d))) %>% 
  ungroup() %>% 
  arrange(p.value)
