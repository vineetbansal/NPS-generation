setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = F)
library(tidyverse)
library(magrittr)
library(broom)
source("R/theme.R")

# read summary
dat = readRDS("data/summary/generalization/evaluate-generalization-gdb13.rds") %>% 
  # remove unused columns
  map(~ {
    keep = map_lgl(.x, ~ n_distinct(.x) > 1)
    extract(.x, , keep)
  })

###############################################################################-
## a. total valid molecules ####
###############################################################################-

p1 = dat$summary %>% 
  filter(outcome == 'total molecules') %>% 
  ggplot(aes(x = representation, y = value, fill = representation,
             color = representation)) +
  facet_wrap(~ 'Novel, valid molecules') +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_fill_manual('', values = representation_pal) +
  scale_color_manual('', values = representation_pal) +
  scale_y_continuous(expression('# of molecules'~(10^6)), 
                     labels = ~ . / 1e6, breaks = pretty_breaks(),
                     limits = c(NA, 1e8)) +
  boxed_theme() +
  theme(aspect.ratio = 1.7,
        legend.position = 'none',
        axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1))
p1
ggsave("fig/fig5/total-valid-molecules.pdf", p1,
       width = 5, height = 5, units = "cm", useDingbats = FALSE)

# test
dat$summary %>% 
  filter(outcome == 'total molecules') %>% 
  group_by(sample_idx) %>% 
  summarise(delta = value[representation == 'SMILES'] - 
              value[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  do(tidy(t.test(.$delta)))

###############################################################################-
## b. molecules not in GDB ####
###############################################################################-

p2 = dat$summary %>% 
  filter(outcome == 'molecules not in GDB') %>% 
  ggplot(aes(x = representation, y = value, fill = representation,
             color = representation)) +
  facet_wrap(~ 'Molecules not in GDB') +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_fill_manual('', values = representation_pal) +
  scale_color_manual('', values = representation_pal) +
  scale_y_continuous(expression('# of molecules'~(10^6)), 
                     labels = ~ . / 1e6, breaks = pretty_breaks()) +
  boxed_theme() +
  theme(aspect.ratio = 1.7,
        legend.position = 'none',
        axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1))
p2
ggsave("fig/fig5/total-molecules-not-in-GDB.pdf", p2,
       width = 5, height = 5, units = "cm", useDingbats = FALSE)

# test
dat$summary %>% 
  filter(outcome == 'molecules not in GDB') %>% 
  group_by(sample_idx) %>% 
  summarise(delta = value[representation == 'SMILES'] - 
              value[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  do(tidy(t.test(.$delta)))

###############################################################################-
## c. molecules in GDB ####
###############################################################################-

p3 = dat$summary %>% 
  filter(outcome == 'molecules in GDB') %>% 
  ggplot(aes(x = representation, y = value, fill = representation,
             color = representation)) +
  facet_wrap(~ 'Molecules in GDB') +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_fill_manual('', values = representation_pal) +
  scale_color_manual('', values = representation_pal) +
  scale_y_continuous(expression('# of molecules'~(10^6)), 
                     labels = ~ . / 1e6, breaks = pretty_breaks()) +
  boxed_theme() +
  theme(aspect.ratio = 1.7,
        legend.position = 'none',
        axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1))
p3
ggsave("fig/fig5/total-molecules-in-GDB.pdf", p3,
       width = 5, height = 5, units = "cm", useDingbats = FALSE)

dat$summary %>% 
  filter(outcome == 'molecules in GDB') %>% 
  group_by(sample_idx) %>% 
  summarise(delta = value[representation == 'SMILES'] - 
              value[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  do(tidy(t.test(.$delta)))

###############################################################################-
## d. percentage of GDB-13 curve ####
###############################################################################-

# plot curves: SMILES vs. SELFIES
pal = colors5l[c(2, 5)] %>% rev
p4 = dat$curves %>% 
  arrange(representation, sample_idx, n_molecules) %>%
  ggplot(aes(x = n_molecules, y = pct_of_gdb, color = representation, 
             fill = representation, group = paste(representation, sample_idx))) +
  geom_path() +
  scale_y_continuous('% of GDB', labels = ~ . * 100) +
  scale_x_continuous(expression('# of sampled molecules'~(10^6)),
                     labels = ~ . / 1e6) +
  scale_fill_manual('', values = representation_pal,
                    breaks = c('SMILES', 'SELFIES'),
                    limits = c('SMILES', 'SELFIES')) +
  scale_color_manual('', values = representation_pal,
                     breaks = c('SMILES', 'SELFIES'),
                     limits = c('SMILES', 'SELFIES')) +
  guides(color = guide_legend(override.aes = list(size = 1.2))) +
  boxed_theme() +
  theme(aspect.ratio = 0.9,
        legend.key.width = unit(0.35, 'lines'),
        legend.key.height = unit(0.4, 'lines'),
        legend.position = c(0.03, 1.03),
        legend.justification = c(0, 1))
p4
ggsave("fig/fig5/unique-molecules-curve.pdf", p4,
       width = 5, height = 5, units = "cm", useDingbats = FALSE)
