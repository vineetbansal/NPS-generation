setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = F)
library(tidyverse)
library(magrittr)
library(broom)
source("R/theme.R")

# read outcome distributions for an example training set
## note: these are all for default parameters, sample_idx=1
train = read_csv("data/summary/outcome-distrs/outcome-distrs-train.csv.gz")
smiles = read_csv("data/summary/outcome-distrs/outcome-distrs-SMILES.csv.gz")
selfies = read_csv("data/summary/outcome-distrs/outcome-distrs-SELFIES.csv.gz")

# combine
dat = bind_rows(
  train %>% mutate(xval = 'Training set'),
  smiles %>% mutate(xval = 'SMILES'),
  selfies %>% mutate(xval = 'SELFIES')
) %>% 
  mutate(xval = fct_relevel(xval, 'Training set', 'SMILES', 'SELFIES'))

# read effect sizes
effs = readRDS("data/summary/outcome-distrs/effect-sizes.rds") %>% 
  map(~ mutate(.x, representation = fct_relevel(representation, 
                                                'SMILES', 'SELFIES')))

###############################################################################-
## a. aromatic rings, example ####
###############################################################################-

aro = filter(dat, outcome == '# of aromatic rings') %>% 
  # winsorize
  mutate(value = ifelse(value >= 5, 5, value))

# convert to proportions
counts = dplyr::count(aro, xval, value) %>% 
  # convert to proportions
  group_by(xval) %>% 
  mutate(prop = n / sum(n)) %>% 
  ungroup() %>% 
  mutate(value = factor(value, levels = seq(0, 10)) %>% 
           fct_rev())

# set up plot data
wide = counts %>% 
  dplyr::select(xval, value, prop) %>% 
  spread(xval, prop) %>% 
  mutate_all(~ replace(., is.na(.), 0))
colnames = colnames(wide)[-1] %>% head(-1)
polygons = map_dfr(seq_along(colnames), ~ {
  colname1 = colnames(wide)[-1][.x]
  colname2 = colnames(wide)[-1][.x + 1]
  print(colname1)
  print(colname2)
  bind_rows(
    mutate(wide, id = value,
           x = .x + 0.25, y = c(0, cumsum(wide[[colname1]]) %>% head(-1))),
    mutate(wide, id = value,
           x = .x + 0.25, y = c(cumsum(wide[[colname1]]))),
    mutate(wide, id = value,
           x = .x + 0.75, y = c(cumsum(wide[[colname2]]))),
    mutate(wide, id = value,
           x = .x + 0.75, y = c(0, cumsum(wide[[colname2]]) %>% head(-1)))
  ) %>% 
    mutate(xval = as.character(.x))
}) %>% 
  mutate(y = 1 - y)

# plot
ring_pal = pals::kovesi.diverging_bwr_40_95_c42(6) %>% rev
p1 = polygons %>% 
  ggplot(aes(x = x, y = y, value = value, fill = value)) +
  geom_polygon(alpha = 0.4, color = 'white', size = 0.13, 
               aes(group = paste(value, xval))) +
  geom_col(data = counts, aes(x = as.integer(xval), y = prop),
           color = 'grey15', size = 0.15, width = 0.5) +
  geom_segment(aes(y = 0, yend = 1, x = -Inf, xend = -Inf), size = 0.4,
               color = 'grey50') +
  scale_x_continuous(breaks = seq_along(unique(counts$xval)),
                     labels = colnames(wide)[-1],
                     limits = c(0.5, 7.5)) +
  scale_y_continuous('% of molecules', labels = function(x) x * 100,
                     limits = c(0, 1.001)) +
  scale_fill_manual(values = ring_pal, name = '# of aromatic rings',
                    labels = ~ replace(., . == '5', '5+')) +
  clean_theme() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.ticks.length.x = unit(-0.2, 'lines'),
        axis.line.x = element_blank(),
        axis.line.y = element_blank(),
        aspect.ratio = 1.2,
        legend.key.size = unit(0.3, 'lines'),
        legend.position = 'right')
p1
ggsave("fig/fig4/aromatic-rings-example-sankey.pdf", p1,
       width = 7, height = 4.5, units = "cm", useDingbats = FALSE)

# also plot total number of rings
ali = filter(dat, outcome == '# of aliphatic rings')
rings = aro %>% 
  mutate(aromatic = value) %>% 
  cbind(aliphatic = ali$value) %>% 
  mutate(value = aromatic + aliphatic) %>%  
  mutate(value = ifelse(value >= 5, 5, value))

###############################################################################-
## b. aromatic rings, effect sizes ####
###############################################################################-

eff = effs$continuous %>% filter(outcome == '# of aromatic rings')

# test p-values
pvals = eff %>% 
  ungroup() %>% 
  dplyr::select(database, representation, n_molecules, min_tc, sample_idx,
                eff) %>%
  distinct() %>% 
  group_by(database, n_molecules, min_tc, sample_idx) %>% 
  filter(n_distinct(representation) == 2) %>% 
  summarise(delta = eff[representation == 'SMILES'] - 
           eff[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  do(tidy(t.test(.$delta))) %>% 
  mutate(label = ifelse(p.value < 0.001, '***',
                        ifelse(p.value < 0.01, '**',
                               ifelse(p.value < 0.05, '*', ''))))
pvals

# boxplot
rep_pal = colors5l[c(2, 5)]
subtitle = paste0('p = ', format(pvals$p.value, digits = 2))
p2 = eff %>% 
  ungroup() %>% 
  distinct(representation, sample_idx, eff) %>% 
  ggplot(aes(x = representation, y = -eff, color = representation,
             fill = representation)) +
  facet_wrap(~ 'Aromatic rings') +
  geom_hline(aes(yintercept = 0), color = 'grey88', size = 0.45) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous('Effect size') +
  scale_color_manual('', values = representation_pal) +
  scale_fill_manual('', values = representation_pal) +
  boxed_theme() +
  theme(legend.position = 'none',
        strip.text = element_text(size = 6),
        axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        plot.subtitle = element_text(size = 5, hjust = 0.5),
        aspect.ratio = 1.7)
p2
ggsave("fig/fig4/aromatic-rings-effect-size-boxplot.pdf", p2, 
       width = 5, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## c. aliphatic rings, example ####
###############################################################################-

ali = filter(dat, outcome == '# of aliphatic rings') %>% 
  # winsorize
  mutate(value = ifelse(value >= 5, 5, value))

# convert to proportions
counts = dplyr::count(ali, xval, value) %>% 
  # convert to proportions
  group_by(xval) %>% 
  mutate(prop = n / sum(n)) %>% 
  ungroup() %>% 
  mutate(value = factor(value, levels = seq(0, 10)) %>% 
           fct_rev())

# set up plot data
wide = counts %>% 
  dplyr::select(xval, value, prop) %>% 
  spread(xval, prop) %>% 
  mutate_all(~ replace(., is.na(.), 0))
colnames = colnames(wide)[-1] %>% head(-1)
polygons = map_dfr(seq_along(colnames), ~ {
  colname1 = colnames(wide)[-1][.x]
  colname2 = colnames(wide)[-1][.x + 1]
  print(colname1)
  print(colname2)
  bind_rows(
    mutate(wide, id = value,
           x = .x + 0.25, y = c(0, cumsum(wide[[colname1]]) %>% head(-1))),
    mutate(wide, id = value,
           x = .x + 0.25, y = c(cumsum(wide[[colname1]]))),
    mutate(wide, id = value,
           x = .x + 0.75, y = c(cumsum(wide[[colname2]]))),
    mutate(wide, id = value,
           x = .x + 0.75, y = c(0, cumsum(wide[[colname2]]) %>% head(-1)))
  ) %>% 
    mutate(xval = as.character(.x))
}) %>% 
  mutate(y = 1 - y)

# plot
p3 = polygons %>% 
  ggplot(aes(x = x, y = y, value = value, fill = value)) +
  geom_polygon(alpha = 0.4, color = 'white', size = 0.13, 
               aes(group = paste(value, xval))) +
  geom_col(data = counts, aes(x = as.integer(xval), y = prop),
           color = 'grey15', size = 0.15, width = 0.5) +
  geom_segment(aes(y = 0, yend = 1, x = -Inf, xend = -Inf), size = 0.4,
               color = 'grey50') +
  scale_x_continuous(breaks = seq_along(unique(counts$xval)),
                     labels = colnames(wide)[-1],
                     limits = c(0.5, 7.5)) +
  scale_y_continuous('% of molecules', labels = function(x) x * 100,
                     limits = c(0, 1.001)) +
  scale_fill_manual(values = ring_pal, name = '# of aliphatic rings',
                    labels = ~ replace(., . == '5', '5+')) +
  clean_theme() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.ticks.length.x = unit(-0.2, 'lines'),
        axis.line.x = element_blank(),
        axis.line.y = element_blank(),
        aspect.ratio = 1.2,
        legend.key.size = unit(0.3, 'lines'),
        legend.position = 'right')
p3
ggsave("fig/fig4/aliphatic-rings-example-sankey.pdf", p3,
       width = 7, height = 4.5, units = "cm", useDingbats = FALSE)

###############################################################################-
## d. aliphatic rings, effect sizes ####
###############################################################################-

eff = effs$continuous %>% filter(outcome == '# of aliphatic rings')

# test p-values
pvals = eff %>% 
  ungroup() %>% 
  dplyr::select(database, representation, n_molecules, min_tc, sample_idx,
                eff) %>%
  distinct() %>% 
  group_by(database, n_molecules, min_tc, sample_idx) %>% 
  filter(n_distinct(representation) == 2) %>% 
  summarise(delta = eff[representation == 'SMILES'] - 
           eff[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  do(tidy(t.test(.$delta))) %>% 
  mutate(label = ifelse(p.value < 0.001, '***',
                        ifelse(p.value < 0.01, '**',
                               ifelse(p.value < 0.05, '*', ''))))
pvals

# boxplot
rep_pal = colors5l[c(2, 5)]
subtitle = paste0('p = ', format(pvals$p.value, digits = 2))
p4 = eff %>% 
  ungroup() %>% 
  distinct(representation, sample_idx, eff) %>% 
  ggplot(aes(x = representation, y = -eff, color = representation,
             fill = representation)) +
  facet_wrap(~ "Aliphatic rings") +
  geom_hline(aes(yintercept = 0), color = 'grey88', size = 0.45) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous('Effect size') +
  scale_color_manual('', values = representation_pal) +
  scale_fill_manual('', values = representation_pal) +
  boxed_theme() +
  theme(legend.position = 'none',
        strip.text = element_text(size = 6),
        axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        plot.subtitle = element_text(size = 5, hjust = 0.5),
        aspect.ratio = 1.7)
p4
ggsave("fig/fig4/aliphatic-rings-effect-size-boxplot.pdf", p2, 
       width = 5, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## e. 'meta-volcano' plot ####
###############################################################################-

continuous_outcomes = unique(effs$continuous$outcome) %>% 
  extract(!grepl('QED|Synthetic|Natural', .))
volcano = map_dfr(continuous_outcomes, ~ {
  continuous_outcome = .x
  # get that outcome
  eff = effs$continuous %>% filter(outcome == continuous_outcome)
  
  # test p-values
  pvals = eff %>% 
    ungroup() %>% 
    dplyr::select(database, representation, n_molecules, min_tc, sample_idx, 
                  eff) %>%
    distinct() %>% 
    group_by(database, n_molecules, min_tc, sample_idx) %>% 
    filter(n_distinct(representation) == 2) %>% 
    summarise(delta = -eff[representation == 'SMILES'] - 
                -eff[representation == 'SELFIES']) %>% 
    ungroup() %>% 
    do(tidy(t.test(.$delta))) %>% 
    mutate(outcome = continuous_outcome)
}) %>% arrange(p.value)

max = max(abs(volcano$estimate))
range = c(-max, max)
p5 = volcano %>% 
  filter(!grepl('Natural product|QED|Synthetic access', outcome)) %>% 
  mutate(outcome = fct_recode(outcome, 
                              'Topological complexity' = 'BertzTC') %>% 
           as.character() %>% 
           gsub("# of ", "", .) %>% 
           Hmisc::capitalize()) %>% 
  ggplot(aes(x = estimate, y = -log10(p.value))) +
  geom_vline(aes(xintercept = 0), color = 'grey88', size = 0.45) +
  geom_hline(aes(yintercept = 1), linetype = 'dotted', size = 0.2) +
  geom_point(shape = 21, size = 0.9, stroke = 0.25, color = 'black',
             fill = 'grey88') +
  geom_text_repel(aes(label = outcome), size = 1.5, min.segment.length = 0, 
                  segment.size = 0.15) +
  scale_x_continuous(expression(Delta[SMILES~-~SELFIES]), limits = range) +
  scale_y_continuous(expression(-log[10](P))) +
  boxed_theme() +
  theme(aspect.ratio = 1)
p5
ggsave("fig/fig4/meta-volcano.pdf", p5, 
       width = 5, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## f. aromatic rings, effect sizes, valid vs. invalid ####
###############################################################################-

# read summary
effs2 = readRDS("data/summary/outcome-distrs/effect-sizes-break-SELFIES.rds") %>% 
  # filter to default params 
  map(~ filter(.x, constraints == 'none') %>% 
        mutate(subset = str_to_title(subset) %>%
                 fct_relevel('Invalid', 'Valid')))

# plot
eff = effs2$continuous %>% filter(outcome == '# of aromatic rings')

# test p-values
pvals = eff %>% 
  ungroup() %>% 
  dplyr::select(database, subset, n_molecules, min_tc, sample_idx, eff) %>%
  distinct() %>% 
  group_by(database, n_molecules, min_tc, sample_idx) %>% 
  filter(n_distinct(subset) == 2) %>% 
  summarise(delta = eff[subset == 'Valid'] - 
              eff[subset == 'Invalid']) %>% 
  ungroup() %>% 
  do(tidy(t.test(.$delta))) %>% 
  mutate(label = ifelse(p.value < 0.001, '***',
                        ifelse(p.value < 0.01, '**',
                               ifelse(p.value < 0.05, '*', ''))))
pvals

# boxplot
p6 = eff %>% 
  ungroup() %>% 
  distinct(subset, sample_idx, eff) %>% 
  ggplot(aes(x = subset, y = -eff, color = subset, fill = subset)) +
  facet_wrap(~ 'Aromatic rings') +
  geom_hline(aes(yintercept = 0), color = 'grey88', size = 0.45) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous('Effect size', breaks = pretty_breaks()) +
  scale_color_manual('', values = valid_pal) +
  scale_fill_manual('', values = valid_pal) +
  boxed_theme() +
  theme(legend.position = 'none',
        strip.text = element_text(size = 6),
        axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        plot.subtitle = element_text(size = 5, hjust = 0.5),
        aspect.ratio = 1.7)
p6
ggsave("fig/fig4/aromatic-rings-effect-size-boxplot-valid-vs-invalid.pdf", p6, 
       width = 5, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## g. aliphatic rings, effect sizes, valid vs. invalid ####
###############################################################################-

eff = effs2$continuous %>% filter(outcome == '# of aliphatic rings')

# test p-values
pvals = eff %>% 
  ungroup() %>% 
  dplyr::select(database, subset, n_molecules, min_tc, sample_idx, eff) %>%
  distinct() %>% 
  group_by(database, n_molecules, min_tc, sample_idx) %>% 
  filter(n_distinct(subset) == 2) %>% 
  summarise(delta = eff[subset == 'Valid'] - 
              eff[subset == 'Invalid']) %>% 
  ungroup() %>% 
  do(tidy(t.test(.$delta))) %>% 
  mutate(label = ifelse(p.value < 0.001, '***',
                        ifelse(p.value < 0.01, '**',
                               ifelse(p.value < 0.05, '*', ''))))
pvals

# boxplot
p7 = eff %>% 
  ungroup() %>% 
  distinct(subset, sample_idx, eff) %>% 
  ggplot(aes(x = subset, y = -eff, color = subset, fill = subset)) +
  facet_wrap(~ 'Aliphatic rings') +
  geom_hline(aes(yintercept = 0), color = 'grey88', size = 0.45) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous('Effect size') +
  scale_color_manual('', values = valid_pal) +
  scale_fill_manual('', values = valid_pal) +
  boxed_theme() +
  theme(legend.position = 'none',
        strip.text = element_text(size = 6),
        axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        plot.subtitle = element_text(size = 5, hjust = 0.5),
        aspect.ratio = 1.7)
p7
ggsave("fig/fig4/aliphatic-rings-effect-size-boxplot-valid-vs-invalid.pdf", p7, 
       width = 5, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## h. correlation, effect sizes, valid-invalid vs. SMILES-SELFIES ####
###############################################################################-

# get break-SELFIES volcano
brk_volcano = map_dfr(continuous_outcomes, ~ {
  continuous_outcome = .x
  # get that outcome
  brk = effs2$continuous %>% filter(outcome == continuous_outcome)
  
  # test p-values
  pvals = brk %>% 
    ungroup() %>% 
    dplyr::select(database, subset, n_molecules, min_tc, sample_idx, eff) %>%
    distinct() %>% 
    group_by(database, n_molecules, min_tc, sample_idx) %>% 
    filter(n_distinct(subset) == 2) %>% 
    summarise(delta = -eff[subset == 'Valid'] - 
             -eff[subset == 'Invalid']) %>% 
    ungroup() %>% 
    do(tidy(t.test(.$delta))) %>% 
    mutate(outcome = continuous_outcome)
}) %>% arrange(p.value)

# compare
xy = left_join(volcano, brk_volcano, by = 'outcome')
cor = cor.test(xy$estimate.x, xy$estimate.y) %>% 
  tidy() %>% 
  mutate(label = paste0('p = ', format(p.value, format = 'f', digits = 2),
                        '\nr = ', formatC(estimate, format = 'f', digits = 2)))
p8 = xy %>% 
  ggplot(aes(x = estimate.x, y = estimate.y)) +
  geom_vline(aes(xintercept = 0), color = 'grey88', size = 0.45) +
  geom_hline(aes(yintercept = 0), color = 'grey88', size = 0.45) +
  geom_point(shape = 21, size = 0.9, stroke = 0.25, color = 'black',
             fill = 'grey88') +
  geom_smooth(method = 'lm', color = 'red', size = 0.25, alpha = 0.15) +
  geom_label(data = cor, aes(label = label, x = -Inf, y = Inf), size = 1.75,
             label.padding = unit(0.6, 'lines'), label.size = NA,
             vjust = 1, hjust = 0, fill = NA) +
  scale_x_continuous(expression(Delta[SMILES~-~SELFIES])) +
  scale_y_continuous(expression(Delta[Valid~-~Invalid])) +
  boxed_theme() +
  theme(aspect.ratio = 1)
p8
ggsave("fig/fig4/meta-volcano-comparison.pdf", p8, 
       width = 6, height = 6, units = "cm", useDingbats = FALSE)
