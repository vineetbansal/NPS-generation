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
## a. LOTUS, SELFIES vs. SMILES vs. training set, top-1 Tc ####
###############################################################################-

df1 = readRDS("data/summary/prior/Tc-all-LOTUS.rds") %>%
  filter(target_source %in% c('model', 'train')) %>%  
  mutate(xval = ifelse(target_source == 'train', 
                       'Training set', representation) %>% 
           fct_relevel('SMILES', 'Training set', 'SELFIES', 
                       'Unconstrained SELFIES')) %>% 
  filter(xval != 'Unconstrained SELFIES')

labs1 = df1 %>% 
  group_by(xval) %>% 
  summarise(mean = mean(Tc), median = median(Tc)) %>% 
  ungroup() %>% 
  mutate(label = format(mean, digits = 2))
p1 = df1 %>% 
  ggplot(aes(x = xval, y = Tc, 
             color = xval, fill = xval)) +
  geom_violin(alpha = 0.25, width = 0.5, show.legend = FALSE, color = NA) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) +
  geom_text(data = labs1, aes(y = -0.05, label = label), size = 1.75, vjust = 0,
            show.legend = FALSE) +
  scale_x_discrete(labels = ~ chartr(' ', '\n', .)) +
  scale_y_continuous('Tanimoto coefficient', breaks = pretty_breaks()) +
  coord_cartesian(ylim = c(-0.05, 0.5)) +
  scale_color_manual('', values = c(representation_pal, prior_pal)) +
  scale_fill_manual('', values = c(representation_pal, prior_pal)) +
  boxed_theme() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, lineheight = 0.9),
        axis.title.x = element_blank(),
        legend.key.width = unit(0.4, 'lines'),
        legend.position = 'none',
        aspect.ratio = 1.5)
p1
ggsave("fig/supp-prior-SELFIES/Tc-LOTUS-training-set-SELFIES.pdf", p1, 
       width = 5, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## b. top-1/top-k, COCONUT, vs. SELFIES/unconstrained SELFIES ####
###############################################################################-

df2 = filter(topk0, database == 'COCONUT',
             target_source == 'model') %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES', 
                                      'Unconstrained SELFIES'))

p2a = df2 %>% 
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
p2a

p2b = df2 %>% 
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
p2b

p2 = p2a | p2b
p2
ggsave("fig/supp-prior-SELFIES/accuracy-COCONUT-vs-SELFIES.pdf", p2, 
       width = 10, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## c. Tc, COCONUT, vs. SELFIES/unconstrained SELFIES ####
###############################################################################-

df3 = readRDS("data/summary/prior/Tc-all-COCONUT.rds") %>%
  filter(target_source == 'model') %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES', 
                                      'Unconstrained SELFIES'))

labs3 = df3 %>% 
  group_by(representation) %>% 
  summarise(mean = mean(Tc), median = median(Tc)) %>% 
  ungroup() %>% 
  mutate(label = format(mean, digits = 2))
p3 = df3 %>% 
  ggplot(aes(x = representation, y = Tc, 
             color = representation, fill = representation)) +
  geom_violin(alpha = 0.25, width = 0.5, show.legend = FALSE, color = NA) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) +
  # geom_jitter(size = 0.1, shape = 21, height = 0, width = 0.6) +
  geom_text(data = labs3, aes(y = -0.04, label = label), size = 1.75, vjust = 0,
            show.legend = FALSE) +
  scale_x_discrete(labels = ~ chartr(' ', '\n', .)) +
  scale_y_continuous('Tanimoto coefficient', breaks = pretty_breaks()) +
  coord_cartesian(ylim = c(-0.04, 0.32)) +
  scale_color_manual('', values = representation_pal) +
  scale_fill_manual('', values = representation_pal) +
  boxed_theme() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, lineheight = 0.9),
        axis.title.x = element_blank(),
        legend.key.width = unit(0.4, 'lines'),
        legend.position = 'none',
        aspect.ratio = 1.5)
p3
ggsave("fig/supp-prior-SELFIES/Tc-COCONUT-vs-SELFIES.pdf", p3, 
       width = 5, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## d. number of candidates, COCONUT, vs. SELFIES/unconstrained SELFIES ####
###############################################################################-

df4 = cands %>% 
  filter(database == 'COCONUT', target_source == 'model') %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES', 
                                      'Unconstrained SELFIES')) %>% 
  filter(!grepl('f', cv_fold))
keep = map_lgl(df4, ~ n_distinct(.x) > 1)
df4 %<>% as.data.frame() %>% extract(, keep)

labs4 = df4 %>% 
  group_by(representation) %>% 
  summarise(mean = mean(n_candidates), median = median(n_candidates)) %>% 
  ungroup() %>% 
  mutate(label = format(round(mean), big.mark = ','))
p4 = df4 %>% 
  ggplot(aes(x = representation, y = n_candidates, 
             color = representation, fill = representation)) +
  geom_violin(alpha = 0.25, width = 0.5, show.legend = FALSE, color = NA) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) +
  # geom_jitter(size = 0.1, shape = 21, height = 0, width = 0.6) +
  geom_label(data = labs4, aes(y = 0, label = label), size = 1.75, vjust = 1,
             show.legend = FALSE, label.padding = unit(0.6, 'lines'),
             label.size = NA, fill = NA) +
  scale_x_discrete(labels = ~ chartr(' ', '\n', .)) +
  scale_y_continuous(expression('# of candidate structures'~(10^3)),
                     breaks = pretty_breaks(), labels = ~ . / 1e3) +
  coord_cartesian(ylim = c(-3000, 30e3)) +
  scale_color_manual('', values = representation_pal) +
  scale_fill_manual('', values = representation_pal) +
  boxed_theme() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, lineheight = 0.9),
        axis.title.x = element_blank(),
        legend.key.width = unit(0.4, 'lines'),
        legend.position = 'none',
        aspect.ratio = 1.5)
p4
ggsave("fig/supp-prior-SELFIES/n-candidates-COCONUT-vs-SELFIES.pdf", p4, 
       width = 5, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## e. top-1/top-k, FooDB, vs. SELFIES/unconstrained SELFIES ####
###############################################################################-

df5 = filter(topk0, database == 'FooDB',
             target_source == 'model') %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES', 
                                      'Unconstrained SELFIES'))

p5a = df5 %>% 
  filter(rank == 1) %>% 
  ggplot(aes(x = representation, y = mean, 
             fill = representation, color = representation)) +
  geom_col(aes(y = Inf), fill = 'grey96', color = NA, width = 0.75) +
  geom_col(aes(y = mean), width = 0.75) +
  geom_errorbar(aes(ymin = mean, ymax = mean + sd), size = 0.35, width = 0) +
  geom_label(aes(y = Inf, label = round(100 * mean, digits = 1) %>% paste0('%')), 
             size = 1.5, vjust = 1, label.padding = unit(0.6, 'lines'),
             fill = NA, label.size = NA) +
  scale_x_discrete(labels = ~ chartr(' ', '\n', .)) +
  scale_y_continuous('Top-1 accuracy (%)', expand = c(0, 0),
                     labels = ~ . * 100) +
  scale_fill_manual('', values = representation_pal) +
  scale_color_manual('', values = representation_pal) +
  coord_cartesian(ylim = c(0, 0.25)) +
  boxed_theme() +
  theme(aspect.ratio = 1.8,
        axis.text.x = element_text(angle = 45, hjust = 1, lineheight = 0.9),
        axis.title.x = element_blank(),
        legend.position = 'none')
p5a

p5b = df5 %>% 
  filter(rank <= 30) %>% 
  ggplot(aes(x = rank, y = mean, 
             fill = representation, color = representation)) +
  geom_stepribbon(aes(ymin = mean - sd, ymax = mean + sd), 
                  alpha = 0.2, size = 0, color = NA, show.legend = FALSE) +
  geom_step(size = 0.35) + 
  scale_x_log10('k') +
  scale_y_continuous('Top-k accuracy (%)', labels = ~ . * 100,
                     limits = c(NA, NA)) +
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
p5b

p5 = p5a | p5b
p5
ggsave("fig/supp-prior-SELFIES/accuracy-FooDB-vs-SELFIES.pdf", p5, 
       width = 10, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## f. Tc, FooDB, vs. SELFIES/unconstrained SELFIES ####
###############################################################################-

df6 = readRDS("data/summary/prior/Tc-all-FooDB.rds") %>%
  filter(target_source == 'model') %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES', 
                                      'Unconstrained SELFIES'))

labs6 = df6 %>% 
  group_by(representation) %>% 
  summarise(mean = mean(Tc), median = median(Tc)) %>% 
  ungroup() %>% 
  mutate(label = format(mean, digits = 2))
p6 = df6 %>% 
  ggplot(aes(x = representation, y = Tc, 
             color = representation, fill = representation)) +
  geom_violin(alpha = 0.25, width = 0.5, show.legend = FALSE, color = NA) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) +
  # geom_jitter(size = 0.1, shape = 21, height = 0, width = 0.6) +
  geom_text(data = labs6, aes(y = -0.1, label = label), size = 1.75, vjust = 0,
            show.legend = FALSE) +
  scale_x_discrete(labels = ~ chartr(' ', '\n', .)) +
  scale_y_continuous('Tanimoto coefficient', breaks = pretty_breaks()) +
  coord_cartesian(ylim = c(-0.1, 1)) +
  scale_color_manual('', values = representation_pal) +
  scale_fill_manual('', values = representation_pal) +
  boxed_theme() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, lineheight = 0.9),
        axis.title.x = element_blank(),
        legend.key.width = unit(0.4, 'lines'),
        legend.position = 'none',
        aspect.ratio = 1.5)
p6
ggsave("fig/supp-prior-SELFIES/Tc-FooDB-vs-SELFIES.pdf", p6, 
       width = 5, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## g. number of candidates, FooDB, vs. SELFIES/unconstrained SELFIES ####
###############################################################################-

df7 = cands %>% 
  filter(database == 'FooDB', target_source == 'model') %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES', 
                                      'Unconstrained SELFIES')) %>% 
  filter(!grepl('f', cv_fold))
keep = map_lgl(df7, ~ n_distinct(.x) > 1)
df7 %<>% as.data.frame() %>% extract(, keep)

labs7 = df7 %>% 
  group_by(representation) %>% 
  summarise(mean = mean(n_candidates), median = median(n_candidates)) %>% 
  ungroup() %>% 
  mutate(label = format(round(mean), big.mark = ','))
p7 = df7 %>% 
  ggplot(aes(x = representation, y = n_candidates, 
             color = representation, fill = representation)) +
  geom_violin(alpha = 0.25, width = 0.5, show.legend = FALSE, color = NA) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) +
  # geom_jitter(size = 0.1, shape = 21, height = 0, width = 0.6) +
  geom_label(data = labs7, aes(y = 0, label = label), size = 1.75, vjust = 1,
             show.legend = FALSE, label.padding = unit(0.6, 'lines'),
             label.size = NA, fill = NA) +
  scale_x_discrete(labels = ~ chartr(' ', '\n', .)) +
  scale_y_continuous(expression('# of candidate structures'),
                     breaks = pretty_breaks()) +
  coord_cartesian(ylim = c(-200, 2.1e3)) +
  scale_color_manual('', values = representation_pal) +
  scale_fill_manual('', values = representation_pal) +
  boxed_theme() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, lineheight = 0.9),
        axis.title.x = element_blank(),
        legend.key.width = unit(0.4, 'lines'),
        legend.position = 'none',
        aspect.ratio = 1.5)
p7
ggsave("fig/supp-prior-SELFIES/n-candidates-FooDB-vs-SELFIES.pdf", p7, 
       width = 5, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## h. top-1/top-k, NORMAN, vs. SELFIES/unconstrained SELFIES ####
###############################################################################-

df8 = filter(topk0, database == 'NORMAN',
             target_source == 'model') %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES', 
                                      'Unconstrained SELFIES'))

p8a = df8 %>% 
  filter(rank == 1) %>% 
  ggplot(aes(x = representation, y = mean, 
             fill = representation, color = representation)) +
  geom_col(aes(y = Inf), fill = 'grey96', color = NA, width = 0.75) +
  geom_col(aes(y = mean), width = 0.75) +
  geom_errorbar(aes(ymin = mean, ymax = mean + sd), size = 0.35, width = 0) +
  geom_label(aes(y = Inf, label = round(100 * mean, digits = 1) %>% paste0('%')), 
             size = 1.5, vjust = 1, label.padding = unit(0.6, 'lines'),
             fill = NA, label.size = NA) +
  scale_x_discrete(labels = ~ chartr(' ', '\n', .)) +
  scale_y_continuous('Top-1 accuracy (%)', expand = c(0, 0),
                     labels = ~ . * 100) +
  scale_fill_manual('', values = representation_pal) +
  scale_color_manual('', values = representation_pal) +
  coord_cartesian(ylim = c(0, 0.17)) +
  boxed_theme() +
  theme(aspect.ratio = 1.8,
        axis.text.x = element_text(angle = 45, hjust = 1, lineheight = 0.9),
        axis.title.x = element_blank(),
        legend.position = 'none')
p8a

p8b = df8 %>% 
  filter(rank <= 30) %>% 
  ggplot(aes(x = rank, y = mean, 
             fill = representation, color = representation)) +
  geom_stepribbon(aes(ymin = mean - sd, ymax = mean + sd), 
                  alpha = 0.2, size = 0, color = NA, show.legend = FALSE) +
  geom_step(size = 0.35) + 
  scale_x_log10('k') +
  scale_y_continuous('Top-k accuracy (%)', labels = ~ . * 100,
                     limits = c(NA, NA)) +
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
p8b

p8 = p8a | p8b
p8
ggsave("fig/supp-prior-SELFIES/accuracy-NORMAN-vs-SELFIES.pdf", p8,
       width = 10, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## i. Tc, NORMAN, vs. SELFIES/unconstrained SELFIES ####
###############################################################################-

df9 = readRDS("data/summary/prior/Tc-all-NORMAN.rds") %>%
  filter(target_source == 'model') %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES', 
                                      'Unconstrained SELFIES'))

labs9 = df9 %>% 
  group_by(representation) %>% 
  summarise(mean = mean(Tc), median = median(Tc)) %>% 
  ungroup() %>% 
  mutate(label = format(mean, digits = 2))
p9 = df9 %>% 
  ggplot(aes(x = representation, y = Tc, 
             color = representation, fill = representation)) +
  geom_violin(alpha = 0.25, width = 0.5, show.legend = FALSE, color = NA) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) +
  geom_text(data = labs9, aes(y = -0.05, label = label), size = 1.75, vjust = 0,
            show.legend = FALSE) +
  scale_x_discrete(labels = ~ chartr(' ', '\n', .)) +
  scale_y_continuous('Tanimoto coefficient', breaks = pretty_breaks()) +
  coord_cartesian(ylim = c(-0.05, 0.55)) +
  scale_color_manual('', values = representation_pal) +
  scale_fill_manual('', values = representation_pal) +
  boxed_theme() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, lineheight = 0.9),
        axis.title.x = element_blank(),
        legend.key.width = unit(0.4, 'lines'),
        legend.position = 'none',
        aspect.ratio = 1.5)
p9
ggsave("fig/supp-prior-SELFIES/Tc-NORMAN-vs-SELFIES.pdf", p9, 
       width = 5, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## j. number of candidates, NORMAN, vs. SELFIES/unconstrained SELFIES ####
###############################################################################-

df10 = cands %>% 
  filter(database == 'NORMAN', target_source == 'model') %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES', 
                                      'Unconstrained SELFIES')) %>% 
  filter(!grepl('f', cv_fold))
keep = map_lgl(df10, ~ n_distinct(.x) > 1)
df10 %<>% as.data.frame() %>% extract(, keep)

labs10 = df10 %>% 
  group_by(representation) %>% 
  summarise(mean = mean(n_candidates), median = median(n_candidates)) %>% 
  ungroup() %>% 
  mutate(label = format(round(mean), big.mark = ','))
p10 = df10 %>% 
  ggplot(aes(x = representation, y = n_candidates, 
             color = representation, fill = representation)) +
  geom_violin(alpha = 0.25, width = 0.5, show.legend = FALSE, color = NA) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) +
  geom_label(data = labs10, aes(y = 0, label = label), size = 1.75, vjust = 1,
             show.legend = FALSE, label.padding = unit(0.6, 'lines'),
             label.size = NA, fill = NA) +
  scale_x_discrete(labels = ~ chartr(' ', '\n', .)) +
  scale_y_continuous(expression('# of candidate structures'),
                     breaks = pretty_breaks()) +
  coord_cartesian(ylim = c(-700, 7.1e3)) +
  scale_color_manual('', values = representation_pal) +
  scale_fill_manual('', values = representation_pal) +
  boxed_theme() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, lineheight = 0.9),
        axis.title.x = element_blank(),
        legend.key.width = unit(0.4, 'lines'),
        legend.position = 'none',
        aspect.ratio = 1.5)
p10
ggsave("fig/supp-prior-SELFIES/n-candidates-NORMAN-vs-SELFIES.pdf", p10, 
       width = 5, height = 5, units = "cm", useDingbats = FALSE)
