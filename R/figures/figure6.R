setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = F)
library(tidyverse)
library(magrittr)
library(broom)
source("R/theme.R")
library(pammtools) ## geom_stepribbon

# read data
ever_gen = readRDS("data/summary/prior/ever-generated.rds")
topk = readRDS("data/summary/prior/top-k-accuracy.rds")
tc = readRDS("data/summary/prior/Tc.rds")
cands = readRDS("data/summary/prior/num-candidates.rds")

# read break SELFIES, too
ever_gen2 = readRDS("data/summary/prior/ever-generated-break-SELFIES.rds") %>%
  mutate(representation = 'Unconstrained SELFIES')
topk2 = readRDS("data/summary/prior/top-k-accuracy-break-SELFIES.rds") %>%
  mutate(representation = 'Unconstrained SELFIES')
tc2 = readRDS("data/summary/prior/Tc-break-SELFIES.rds") %>%
  mutate(representation = 'Unconstrained SELFIES')
cands2 = readRDS("data/summary/prior/num-candidates-break-SELFIES.rds") %>%
  mutate(representation = 'Unconstrained SELFIES')

# combine
ever_gen %<>% bind_rows(ever_gen2)
topk %<>% bind_rows(topk2)
tc %<>% bind_rows(tc2)
cands %<>% bind_rows(cands2)

# average over CV folds
topk0 = topk %>% 
  filter(!grepl('^f', cv_fold)) %>% 
  group_by_at(vars(-cv_fold, -n, -top_k)) %>% 
  summarise(mean = mean(top_k),
            sd = sd(top_k)) %>% 
  ungroup() %>% 
  extract(, map_lgl(., ~ n_distinct(.x) > 1))

###############################################################################-
## b. top-1/top-k, LOTUS, vs. PubChem ####
###############################################################################-

df1 = filter(topk0, database == 'LOTUS', representation == 'SMILES',
             target_source != 'train') %>% 
  mutate(target_source = fct_recode(target_source,
                                    'Language model' = 'model') %>% 
           fct_relevel('Language model', 'PubChem'))

p1a = df1 %>% 
  filter(rank == 1) %>% 
  ggplot(aes(x = target_source, y = mean, 
             fill = target_source, color = target_source)) +
  geom_col(aes(y = Inf), fill = 'grey96', color = NA, width = 0.8) +
  geom_col(aes(y = mean), width = 0.8) +
  geom_errorbar(aes(ymin = mean, ymax = mean + sd), size = 0.35, width = 0) +
  geom_label(aes(y = Inf, label = round(100 * mean, digits = 1) %>% paste0('%')), 
             size = 1.75, vjust = 1, label.padding = unit(0.6, 'lines'),
             fill = NA, label.size = NA) +
  scale_y_continuous('Top-1 accuracy (%)', expand = c(0, 0),
                     labels = ~ . * 100) +
  scale_fill_manual('', values = prior_pal) +
  scale_color_manual('', values = prior_pal) +
  coord_cartesian(ylim = c(0, 0.065)) +
  boxed_theme() +
  theme(aspect.ratio = 2.2,
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.x = element_blank(),
        legend.position = 'none')
p1a

p1b = df1 %>% 
  filter(rank <= 30) %>% 
  ggplot(aes(x = rank, y = mean, 
             fill = target_source, color = target_source)) +
  geom_stepribbon(aes(ymin = mean - sd, ymax = mean + sd), 
                  alpha = 0.2, size = 0, color = NA, show.legend = FALSE) +
  geom_step(size = 0.35) + 
  scale_x_log10('k') +
  scale_y_continuous('Top-k accuracy (%)', labels = ~ . * 100) +
  scale_color_manual('', values = prior_pal) +
  scale_fill_manual('', values = prior_pal) +
  annotation_logticks(sides = 'b', size = 0.2, short = unit(0.06, 'lines'),
                      mid = unit(0.12, 'lines'), long = unit(0.19, 'lines')) +
  boxed_theme() +
  theme(aspect.ratio = 1,
        legend.position = c(0.03, 1.04),
        legend.justification = c(0, 1),
        legend.key.width = unit(0.35, 'lines'),
        legend.key.height = unit(0.45, 'lines'))
p1b

p1 = p1a | p1b
p1
ggsave("fig/fig6/accuracy-LOTUS-vs-PubChem.pdf", p1, 
       width = 10, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## d. Tc, LOTUS, vs. PubChem/random/train ####
###############################################################################-

df2 = readRDS("data/summary/prior/Tc-all-LOTUS.rds") %>% 
  filter(representation == 'SMILES') %>% 
  mutate(target_source = fct_recode(target_source,
                                    'Language model' = 'model',
                                    'Training set' = 'train',
                                    'Language model\n(random)' = 'model (random)') %>% 
           fct_relevel('Language model', 'Training set', 
                       'Language model\n(random)', 'PubChem'))

labs2 = df2 %>% 
  group_by(target_source) %>% 
  summarise(mean = mean(Tc), median = median(Tc)) %>% 
  ungroup() %>% 
  mutate(label = format(mean, digits = 2))
p2 = df2 %>% 
  ggplot(aes(x = target_source, y = Tc, 
             color = target_source, fill = target_source)) +
  geom_violin(alpha = 0.25, width = 0.5, show.legend = FALSE, color = NA) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) +
  # geom_jitter(size = 0.1, shape = 21, height = 0, width = 0.6) +
  geom_text(data = labs2, aes(y = -0.05, label = label), size = 1.75, vjust = 0,
            show.legend = FALSE) +
  scale_y_continuous('Tanimoto coefficient') +
  coord_cartesian(ylim = c(-0.05, 0.5)) +
  scale_color_manual('', values = prior_pal) +
  scale_fill_manual('', values = prior_pal) +
  boxed_theme() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.x = element_blank(),
        legend.key.width = unit(0.4, 'lines'),
        legend.position = 'none',
        aspect.ratio = 1.5)
p2
ggsave("fig/fig6/Tc-LOTUS-vs-PubChem.pdf", p2, 
       width = 5, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## e. top-1/top-k, formula, LOTUS, vs. PubChem ####
###############################################################################-

# read data
topk_f = readRDS("data/summary/prior/top-k-accuracy-formula.rds")

# average over CV folds
topk0_f = topk_f %>% 
  filter(!grepl('^f', cv_fold)) %>% 
  group_by_at(vars(-cv_fold, -n, -top_k)) %>% 
  summarise(mean = mean(top_k),
            sd = sd(top_k)) %>% 
  ungroup() %>% 
  extract(, map_lgl(., ~ n_distinct(.x) > 1))

# extract data for plotting
df3 = filter(topk0_f, database == 'LOTUS', representation == 'SMILES',
             target_source != 'train') %>% 
  mutate(target_source = fct_recode(target_source,
                                    'Language model' = 'model') %>% 
           fct_relevel('Language model', 'PubChem'))

# plot top-1
p3a = df3 %>% 
  filter(rank == 1) %>% 
  ggplot(aes(x = target_source, y = mean, 
             fill = target_source, color = target_source)) +
  geom_col(aes(y = Inf), fill = 'grey96', color = NA, width = 0.8) +
  geom_col(aes(y = mean), width = 0.8) +
  geom_errorbar(aes(ymin = mean, ymax = mean + sd), size = 0.35, width = 0) +
  geom_label(aes(y = Inf, label = round(100 * mean, digits = 1) %>% paste0('%')), 
             size = 1.75, vjust = 1, label.padding = unit(0.6, 'lines'),
             fill = NA, label.size = NA) +
  scale_y_continuous('Top-1 accuracy (%)', expand = c(0, 0),
                     labels = ~ . * 100) +
  scale_fill_manual('', values = prior_pal) +
  scale_color_manual('', values = prior_pal) +
  coord_cartesian(ylim = c(0, 1)) +
  boxed_theme() +
  theme(aspect.ratio = 2.2,
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.x = element_blank(),
        legend.position = 'none')
p3a
# plot top-k
p3b = df3 %>% 
  filter(rank <= 30) %>% 
  ggplot(aes(x = rank, y = mean, 
             fill = target_source, color = target_source)) +
  geom_stepribbon(aes(ymin = mean - sd, ymax = mean + sd), 
                  alpha = 0.2, size = 0, color = NA, show.legend = FALSE) +
  geom_step(size = 0.35) + 
  scale_x_log10('k') +
  scale_y_continuous('Top-k accuracy (%)', labels = ~ . * 100,
                     breaks = pretty_breaks(), limits = c(.2, 1)) +
  scale_color_manual('', values = prior_pal) +
  scale_fill_manual('', values = prior_pal) +
  annotation_logticks(sides = 'b', size = 0.2, short = unit(0.06, 'lines'),
                      mid = unit(0.12, 'lines'), long = unit(0.19, 'lines')) +
  boxed_theme() +
  theme(aspect.ratio = 1,
        legend.position = c(0.975, 0.04),
        legend.justification = c(1, 0),
        legend.key.width = unit(0.35, 'lines'),
        legend.key.height = unit(0.45, 'lines'))
p3b

p3 = p3a | p3b
p3
ggsave("fig/fig6/accuracy-LOTUS-vs-PubChem-formula.pdf", p3, 
       width = 10, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## f. number of formula candidates, LOTUS, vs. PubChem/train ####
###############################################################################-

formulas = readRDS("data/summary/prior/num-formulas.rds") %>% 
  mutate(target_source = fct_recode(target_source,
                                    'Language model' = 'model',
                                    'Training set' = 'train',
                                    'Language model\n(random)' = 'model (random)') %>% 
           fct_relevel('Language model', 'Training set', 
                       'Language model\n(random)', 'PubChem'))
formulas0 = filter(formulas, !grepl('f', cv_fold))
keep = map_lgl(formulas0, ~ n_distinct(.x) > 1)
formulas0 %<>% as.data.frame() %>% extract(, keep)

labs4 = formulas0 %>% 
  group_by(database, target_source) %>% 
  summarise(mean = mean(n_candidates), median = median(n_candidates)) %>% 
  ungroup() %>% 
  mutate(label = round(mean))

# overview plot
p4 = formulas0 %>% 
  filter(database == 'LOTUS') %>% 
  # sample random fraction
  sample_frac(0.01) %>% 
  ggplot(aes(x = target_source, y = n_candidates, 
             fill = target_source, color = target_source)) +
  geom_label(data = labs4 %>% 
               filter(database == 'LOTUS'),
             aes(y = 0, label = label), size = 1.75, vjust = 1,
             show.legend = FALSE, label.padding = unit(0.6, 'lines'),
             label.size = NA, fill = NA) +
  geom_violin(alpha = 0.25, width = 0.5, show.legend = FALSE, color = NA) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) +
  scale_y_continuous('# of candidate formulas') +
  coord_cartesian(ylim = c(-20, 200)) +
  scale_color_manual('', values = prior_pal) +
  scale_fill_manual('', values = prior_pal) +
  boxed_theme() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.x = element_blank(),
        legend.key.width = unit(0.4, 'lines'),
        legend.position = 'none',
        aspect.ratio = 1.5)
p4
ggsave("fig/fig6/n-formulas-LOTUS-vs-PubChem.pdf", p4, 
       width = 5, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## g. top-1/top-k, FooDB, vs. PubChem ####
###############################################################################-

df5 = filter(topk0, database == 'FooDB', representation == 'SMILES',
             target_source != 'train') %>% 
  mutate(target_source = fct_recode(target_source,
                                    'Language model' = 'model') %>% 
           fct_relevel('Language model', 'PubChem'))

p5a = df5 %>% 
  filter(rank == 1) %>% 
  ggplot(aes(x = target_source, y = mean, 
             fill = target_source, color = target_source)) +
  geom_col(aes(y = Inf), fill = 'grey96', color = NA, width = 0.75) +
  geom_col(aes(y = mean), width = 0.75) +
  geom_errorbar(aes(ymin = mean, ymax = mean + sd), size = 0.35, width = 0) +
  geom_label(aes(y = Inf, label = round(100 * mean, digits = 1) %>% 
                   formatC(format = 'f', digits = 1) %>% paste0('%')), 
             size = 1.75, vjust = 1, label.padding = unit(0.6, 'lines'),
             fill = NA, label.size = NA) +
  scale_y_continuous('Top-1 accuracy (%)', expand = c(0, 0),
                     labels = ~ . * 100) +
  scale_fill_manual('', values = prior_pal) +
  scale_color_manual('', values = prior_pal) +
  coord_cartesian(ylim = c(0, 0.22)) +
  boxed_theme() +
  theme(aspect.ratio = 2.2,
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.x = element_blank(),
        legend.position = 'none')
p5a

p5b = df5 %>% 
  filter(rank <= 30) %>% 
  ggplot(aes(x = rank, y = mean, 
             fill = target_source, color = target_source)) +
  geom_stepribbon(aes(ymin = mean - sd, ymax = mean + sd), 
                  alpha = 0.2, size = 0, color = NA, show.legend = FALSE) +
  geom_step(size = 0.35) + 
  scale_x_log10('k') +
  scale_y_continuous('Top-k accuracy (%)', labels = ~ . * 100) +
  scale_color_manual('', values = prior_pal) +
  scale_fill_manual('', values = prior_pal) +
  annotation_logticks(sides = 'b', size = 0.2, short = unit(0.06, 'lines'),
                      mid = unit(0.12, 'lines'), long = unit(0.19, 'lines')) +
  boxed_theme() +
  theme(aspect.ratio = 1,
        legend.position = c(0.03, 1.04),
        legend.justification = c(0, 1),
        legend.key.width = unit(0.35, 'lines'),
        legend.key.height = unit(0.45, 'lines'))
p5b

p5 = p5a | p5b
p5
ggsave("fig/fig6/accuracy-FooDB-vs-PubChem.pdf", p5, 
       width = 10, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## h. Tc, FooDB, vs. PubChem/random/train ####
###############################################################################-

df6 = readRDS("data/summary/prior/Tc-all-FooDB.rds") %>% 
  filter(representation == 'SMILES') %>% 
  mutate(target_source = fct_recode(target_source,
                                    'Language model' = 'model',
                                    'Training set' = 'train',
                                    'Language model\n(random)' = 'model (random)') %>% 
           fct_relevel('Language model', 'Training set', 
                       'Language model\n(random)', 'PubChem'))

labs6 = df6 %>% 
  group_by(target_source) %>% 
  summarise(mean = mean(Tc), median = median(Tc)) %>% 
  ungroup() %>% 
  mutate(label = format(mean, digits = 2))
p6 = df6 %>% 
  ggplot(aes(x = target_source, y = Tc, 
             color = target_source, fill = target_source)) +
  geom_violin(alpha = 0.25, width = 0.5, show.legend = FALSE, color = NA) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) +
  # geom_jitter(size = 0.1, shape = 21, height = 0, width = 0.6) +
  geom_text(data = labs6, aes(y = -0.1, label = label), size = 1.75, vjust = 0,
            show.legend = FALSE) +
  scale_y_continuous('Tanimoto coefficient', breaks = pretty_breaks()) +
  coord_cartesian(ylim = c(-0.1, 1)) +
  scale_color_manual('', values = prior_pal) +
  scale_fill_manual('', values = prior_pal) +
  boxed_theme() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, lineheight = 0.9),
        axis.title.x = element_blank(),
        legend.key.width = unit(0.4, 'lines'),
        legend.position = 'none',
        aspect.ratio = 1.5)
p6
ggsave("fig/fig6/Tc-FooDB-vs-PubChem.pdf", p6, 
       width = 5, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## i. top-1/top-k, NORMAN, vs. PubChem ####
###############################################################################-

df7 = filter(topk0, database == 'NORMAN', representation == 'SMILES',
             target_source != 'train') %>% 
  mutate(target_source = fct_recode(target_source,
                                    'Language model' = 'model') %>% 
           fct_relevel('Language model', 'PubChem'))

p7a = df7 %>% 
  filter(rank == 1) %>% 
  ggplot(aes(x = target_source, y = mean, 
             fill = target_source, color = target_source)) +
  geom_col(aes(y = Inf), fill = 'grey96', color = NA, width = 0.75) +
  geom_col(aes(y = mean), width = 0.75) +
  geom_errorbar(aes(ymin = mean, ymax = mean + sd), size = 0.35, width = 0) +
  geom_label(aes(y = Inf, label = round(100 * mean, digits = 1) %>% 
                   formatC(format = 'f', digits = 1) %>% paste0('%')), 
             size = 1.75, vjust = 1, label.padding = unit(0.6, 'lines'),
             fill = NA, label.size = NA) +
  scale_y_continuous('Top-1 accuracy (%)', expand = c(0, 0),
                     labels = ~ . * 100) +
  scale_fill_manual('', values = prior_pal) +
  scale_color_manual('', values = prior_pal) +
  coord_cartesian(ylim = c(0, 0.17)) +
  boxed_theme() +
  theme(aspect.ratio = 2.2,
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.x = element_blank(),
        legend.position = 'none')
p7a

p7b = df7 %>% 
  filter(rank <= 30) %>% 
  ggplot(aes(x = rank, y = mean, 
             fill = target_source, color = target_source)) +
  geom_stepribbon(aes(ymin = mean - sd, ymax = mean + sd), 
                  alpha = 0.2, size = 0, color = NA, show.legend = FALSE) +
  geom_step(size = 0.35) + 
  scale_x_log10('k') +
  scale_y_continuous('Top-k accuracy (%)', labels = ~ . * 100) +
  scale_color_manual('', values = prior_pal) +
  scale_fill_manual('', values = prior_pal) +
  annotation_logticks(sides = 'b', size = 0.2, short = unit(0.06, 'lines'),
                      mid = unit(0.12, 'lines'), long = unit(0.19, 'lines')) +
  boxed_theme() +
  theme(aspect.ratio = 1,
        legend.position = c(0.03, 1.04),
        legend.justification = c(0, 1),
        legend.key.width = unit(0.35, 'lines'),
        legend.key.height = unit(0.45, 'lines'))
p7b

p7 = p7a | p7b
p7
ggsave("fig/fig6/accuracy-NORMAN-vs-PubChem.pdf", p7, 
       width = 10, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## j. Tc, NORMAN, vs. PubChem/random/train ####
###############################################################################-

df8 = readRDS("data/summary/prior/Tc-all-NORMAN.rds") %>% 
  filter(representation == 'SMILES') %>% 
  mutate(target_source = fct_recode(target_source,
                                    'Language model' = 'model',
                                    'Training set' = 'train',
                                    'Language model\n(random)' = 'model (random)') %>% 
           fct_relevel('Language model', 'Training set', 
                       'Language model\n(random)', 'PubChem'))

labs = df8 %>% 
  group_by(target_source) %>% 
  summarise(mean = mean(Tc), median = median(Tc)) %>% 
  ungroup() %>% 
  mutate(label = format(mean, digits = 2))
p8 = df8 %>% 
  ggplot(aes(x = target_source, y = Tc, 
             color = target_source, fill = target_source)) +
  geom_violin(alpha = 0.25, width = 0.5, show.legend = FALSE, color = NA) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) +
  # geom_jitter(size = 0.1, shape = 21, height = 0, width = 0.6) +
  geom_text(data = labs, aes(y = -0.06, label = label), size = 1.75, vjust = 0,
            show.legend = FALSE) +
  scale_y_continuous('Tanimoto coefficient', breaks = pretty_breaks()) +
  coord_cartesian(ylim = c(-0.06, 0.6)) +
  scale_color_manual('', values = prior_pal) +
  scale_fill_manual('', values = prior_pal) +
  boxed_theme() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, lineheight = 0.9),
        axis.title.x = element_blank(),
        legend.key.width = unit(0.4, 'lines'),
        legend.position = 'none',
        aspect.ratio = 1.5)
p8
ggsave("fig/fig6/Tc-NORMAN-vs-PubChem.pdf", p8, 
       width = 5, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## k. top-1/top-k, LOTUS, vs. SELFIES/unconstrained SELFIES ####
###############################################################################-

df9 = filter(topk0, database == 'LOTUS',
             target_source == 'model') %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES', 
                                      'Unconstrained SELFIES'))

p9a = df9 %>% 
  filter(rank == 1) %>% 
  ggplot(aes(x = representation, y = mean, 
             fill = representation, color = representation)) +
  geom_col(aes(y = Inf), fill = 'grey96', color = NA, width = 0.75) +
  geom_col(aes(y = mean), width = 0.75) +
  geom_errorbar(aes(ymin = mean, ymax = mean + sd), size = 0.35, width = 0) +
  geom_label(aes(y = Inf, label = round(100 * mean, digits = 1) %>% paste0('%')), 
             size = 1.75, vjust = 1, label.padding = unit(0.6, 'lines'),
             fill = NA, label.size = NA) +
  scale_x_discrete(labels = ~ chartr(' ', '\n', .)) +
  scale_y_continuous('Top-1 accuracy (%)', expand = c(0, 0),
                     labels = ~ . * 100) +
  scale_fill_manual('', values = representation_pal) +
  scale_color_manual('', values = representation_pal) +
  coord_cartesian(ylim = c(0, 0.065)) +
  boxed_theme() +
  theme(aspect.ratio = 1.8,
        axis.text.x = element_text(angle = 45, hjust = 1, lineheight = 0.9),
        axis.title.x = element_blank(),
        legend.position = 'none')
p9a

p9b = df9 %>% 
  filter(rank <= 30) %>% 
  ggplot(aes(x = rank, y = mean, 
             fill = representation, color = representation)) +
  geom_stepribbon(aes(ymin = mean - sd, ymax = mean + sd), 
                  alpha = 0.2, size = 0, color = NA, show.legend = FALSE) +
  geom_step(size = 0.35) + 
  scale_x_log10('k') +
  scale_y_continuous('Top-k accuracy (%)', labels = ~ . * 100,
                     limits = c(0, NA)) +
  scale_color_manual('', values = representation_pal) +
  scale_fill_manual('', values = representation_pal) +
  annotation_logticks(sides = 'b', size = 0.2, short = unit(0.06, 'lines'),
                      mid = unit(0.12, 'lines'), long = unit(0.19, 'lines')) +
  boxed_theme() +
  theme(aspect.ratio = 1,
        legend.position = c(0.03, 1.04),
        legend.justification = c(0, 1),
        legend.key.width = unit(0.35, 'lines'),
        legend.key.height = unit(0.45, 'lines'))
p9b

p9 = p9a | p9b
p9
ggsave("fig/fig6/accuracy-LOTUS-vs-SELFIES.pdf", p9, 
       width = 10, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## l. Tc, LOTUS, vs. SELFIES/unconstrained SELFIES ####
###############################################################################-

df10 = readRDS("data/summary/prior/Tc-all-LOTUS.rds") %>%
  filter(target_source == 'model') %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES', 
                                      'Unconstrained SELFIES'))

labs10 = df10 %>% 
  group_by(representation) %>% 
  summarise(mean = mean(Tc), median = median(Tc)) %>% 
  ungroup() %>% 
  mutate(label = format(mean, digits = 2))
p10 = df10 %>% 
  ggplot(aes(x = representation, y = Tc, 
             color = representation, fill = representation)) +
  geom_violin(alpha = 0.25, width = 0.5, show.legend = FALSE, color = NA) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) +
  # geom_jitter(size = 0.1, shape = 21, height = 0, width = 0.6) +
  geom_text(data = labs10, aes(y = -0.05, label = label), size = 1.75, vjust = 0,
            show.legend = FALSE) +
  scale_x_discrete(labels = ~ chartr(' ', '\n', .)) +
  scale_y_continuous('Tanimoto coefficient', breaks = pretty_breaks()) +
  coord_cartesian(ylim = c(-0.05, 0.5)) +
  scale_color_manual('', values = representation_pal) +
  scale_fill_manual('', values = representation_pal) +
  boxed_theme() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, lineheight = 0.9),
        axis.title.x = element_blank(),
        legend.key.width = unit(0.4, 'lines'),
        legend.position = 'none',
        aspect.ratio = 1.5)
p10
ggsave("fig/fig6/Tc-LOTUS-vs-SELFIES.pdf", p10, 
       width = 5, height = 5, units = "cm", useDingbats = FALSE)

# tests
xy = df10 %>% 
  filter(representation %in% c('SMILES', 'SELFIES')) %>% 
  group_by(smiles) %>% 
  filter(n_distinct(representation) == 2) %>% 
  ungroup()
x = xy %>% filter(representation == 'SMILES') %>% pull(Tc)
y = xy %>% filter(representation == 'SELFIES') %>% pull(Tc)
t.test(y - x)
wilcox.test(y - x)

yz = df10 %>% 
  filter(representation %in% c('SELFIES', 'Unconstrained SELFIES')) %>% 
  group_by(smiles) %>% 
  filter(n_distinct(representation) == 2) %>% 
  ungroup()
y = yz %>% filter(representation == 'SELFIES') %>% pull(Tc)
z = yz %>% filter(representation == 'Unconstrained SELFIES') %>% pull(Tc)
t.test(z - y)
wilcox.test(z - y)

###############################################################################-
## m. number of candidates, LOTUS, vs. SELFIES/unconstrained SELFIES ####
###############################################################################-

cands = bind_rows(
  readRDS("data/summary/prior/num-candidates.rds"),
  readRDS("data/summary/prior/num-candidates-break-SELFIES.rds") %>% 
    mutate(representation = 'Unconstrained SELFIES')
) %>% 
  filter(database == 'LOTUS', target_source == 'model') %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES', 
                                      'Unconstrained SELFIES'))
cands0 = filter(cands, !grepl('f', cv_fold))
keep = map_lgl(cands0, ~ n_distinct(.x) > 1)
cands0 %<>% as.data.frame() %>% extract(, keep)

labs11 = cands0 %>% 
  group_by(representation) %>% 
  summarise(mean = mean(n_candidates), median = median(n_candidates)) %>% 
  ungroup() %>% 
  mutate(label = format(round(mean), big.mark = ','))
p11 = cands0 %>% 
  ggplot(aes(x = representation, y = n_candidates, 
             color = representation, fill = representation)) +
  geom_violin(alpha = 0.25, width = 0.5, show.legend = FALSE, color = NA) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) +
  # geom_jitter(size = 0.1, shape = 21, height = 0, width = 0.6) +
  geom_label(data = labs11, aes(y = 0, label = label), size = 1.75, vjust = 1,
             show.legend = FALSE, label.padding = unit(0.6, 'lines'),
             label.size = NA, fill = NA) +
  scale_x_discrete(labels = ~ chartr(' ', '\n', .)) +
  scale_y_continuous(expression('# of candidate structures'~(10^3)),
                     breaks = pretty_breaks(), labels = ~ . / 1e3) +
  coord_cartesian(ylim = c(-5000, 55e3)) +
  scale_color_manual('', values = representation_pal) +
  scale_fill_manual('', values = representation_pal) +
  boxed_theme() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, lineheight = 0.9),
        axis.title.x = element_blank(),
        legend.key.width = unit(0.4, 'lines'),
        legend.position = 'none',
        aspect.ratio = 1.5)
p11
ggsave("fig/fig6/n-candidates-LOTUS-vs-SELFIES.pdf", p11, 
       width = 5, height = 5, units = "cm", useDingbats = FALSE)
