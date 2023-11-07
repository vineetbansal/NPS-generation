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
# combine
outcomes = bind_rows(outcomes1, outcomes2) %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES', 
                                      'Texas SELFIES', 'Unconstrained SELFIES'))

###############################################################################-
## a. FCD, Texas/unconstrained SELFIES ####
###############################################################################-

# extract FCD
fcd = filter(outcomes, grepl('Frechet', outcome))

# plot default
df1 = fcd %>% 
  filter(database == 'ChEMBL',
         n_molecules == 1e5,
         enum_factor == 0,
         min_tc == 0)

# plot
p1 = df1 %>% 
  ggplot(aes(x = representation, y = value, fill = representation, 
             color = representation)) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_x_discrete(labels = ~ gsub(" SELFIES", "\nSELFIES", .)) +
  scale_y_continuous('Frechet ChemNet distance') +
  scale_color_manual(values = representation_pal) +
  scale_fill_manual(values = representation_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1.1)
p1
ggsave("fig/fig3/FCD-default-both.pdf", p6,
       width = 4, height = 4.3, units = "cm", useDingbats = FALSE)

# t-tests
test = df1 %>% 
  group_by(sample_idx) %>% 
  summarise(delta = value[representation == 'Texas SELFIES'] -
           value[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  do(tidy(t.test(.$delta)))
message('FCD, default settings, Texas: p = ', 
        format(test$p.value, digits = 2))
test = df1 %>% 
  group_by(sample_idx) %>% 
  summarise(delta = value[representation == 'Unconstrained SELFIES'] -
           value[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  do(tidy(t.test(.$delta)))
message('FCD, default settings, unconstrained: p = ', 
        format(test$p.value, digits = 2))

###############################################################################-
## b. Loss, unconstrained SELFIES, example model ####
###############################################################################-

# read an example dataset
dat1 = read_csv("data/summary/break-SELFIES/sample-1-constraints=none-valid.csv.gz",
                col_names = c('loss', 'smiles', 'tokens', 'msg', 'error'), 
                skip = 1) %>% 
  mutate(valid = TRUE)
dat2 = read_csv("data/summary/break-SELFIES/sample-1-constraints=none-invalid.csv.gz",
                col_names = c('loss', 'smiles', 'tokens', 'msg', 'error'), 
                skip = 1) %>% 
  mutate(valid = FALSE)
dat = bind_rows(dat1, dat2)

range = boxplot(loss ~ valid, data = dat)$stats %>% range
p2 = dat %>% 
  ggplot(aes(x = valid, y = loss, color = valid, fill = valid)) +
  geom_violin(alpha = 0.15, color = NA, width = 0.7) +
  geom_boxplot(outlier.shape = NA, width = 0.6, alpha = 0.4, size = 0.35) + 
  scale_y_continuous('Loss', expand = c(0.1, 0)) +
  scale_x_discrete(labels = c('FALSE' = 'Invalid\nSELFIES',
                              'TRUE' = 'Valid\nSELFIES')) +
  scale_fill_manual(values = unname(valid_pal)) +
  scale_color_manual(values = unname(valid_pal)) +
  coord_cartesian(ylim = range) +
  boxed_theme() +
  theme(aspect.ratio = 1.7,
        legend.position = 'none',
        axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1))
p2
ggsave("fig/fig3/example-loss-boxplot.pdf", p2,
       width = 5, height = 4, units = "cm", useDingbats = FALSE)

x = filter(dat, valid) %>% pull(loss)
y = filter(dat, !valid) %>% pull(loss)
wilcox.test(x, y)

###############################################################################-
## c. Loss, unconstrained SELFIES, effect sizes ####
###############################################################################-

# read data
eff = readRDS("data/summary/break-SELFIES/parse-valid-SELFIES-RNN.rds")$effects %>% 
  # filter to unconstrained
  filter(constraints == 'none')

# plot default
df3 = eff %>% 
  filter(analysis == 'loss') %>% 
  filter(database == 'ChEMBL',
         representation == 'SELFIES',
         n_molecules == 1e5,
         enum_factor == 0,
         min_tc == 0)
p3 = df3 %>% 
  mutate(xval = 'Valid vs.\ninvalid\nSELFIES') %>% 
  ggplot(aes(x = xval, y = d, fill = representation, color = representation)) +
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
        aspect.ratio = 2 * 1.7)
p3
ggsave("fig/fig3/loss-d-default.pdf", p3,
       width = 4, height = 4, units = "cm", useDingbats = FALSE)

# t-test
test = df3 %>% 
  do(tidy(t.test(.$d)))
message('loss, default settings: p = ', format(test$p.value, digits = 2))

###############################################################################-
## d. % valid vs. delta-FCD ####
###############################################################################-

# read data
valid = readRDS("data/summary/break-SELFIES/SELFIES-percent-valid.rds") %>% 
  mutate(representation = fct_recode(constraints, 
                                     'Texas SELFIES' = 'c5', 
                                     'Unconstrained SELFIES' = 'none')) %>% 
  filter(representation == 'Unconstrained SELFIES')
# do we have all the required data?
valid %>% 
  dplyr::count(database, enum_factor, n_molecules, min_tc) %>% 
  arrange(database, n_molecules, min_tc, enum_factor)
fcd %>% 
  dplyr::count(database, enum_factor, n_molecules, min_tc, representation)

# compute delta-FCD
delta_fcd = fcd %>% 
  filter(representation == 'SELFIES' | constraints == 'none') %>% 
  group_by_at(vars(-constraints, -representation, -value)) %>% 
  summarise(delta_fcd = value[representation != 'SELFIES'] -
              value[representation == 'SELFIES']) %>% 
  ungroup()
# do we have all the required data?
delta_fcd %>% 
  dplyr::count(database, enum_factor, n_molecules, min_tc)

df4 = delta_fcd %>%
  inner_join(valid, by = c('database', 'enum_factor', 'n_molecules', 'min_tc',
                           'sample_idx')) %>% 
  dplyr::select(-starts_with('constraints'), -outcome)

# correlation as legend
library(broom)
cor = df4 %>% 
  group_by(representation) %>% 
  do(tidy(cor.test(.$valid, .$delta_fcd))) %>% 
  mutate(label = paste0('r = ', formatC(estimate, format = 'f', digits = 2), 
                        '\np = ', format(p.value, digits = 2)))

# plot
p4 = df4 %>% 
  ggplot(aes(x = valid, y = delta_fcd)) +
  geom_point(shape = 21, size = 0.9, stroke = 0.25, color = 'black',
             fill = 'grey88') +
  geom_smooth(method = 'lm', color = 'red', size = 0.25, alpha = 0.15) +
  geom_label(data = cor, aes(x = Inf, y = -Inf, label = label), 
             hjust = 1, vjust = 0, size = 1.75, lineheight = 0.95,
             label.padding = unit(0.7, 'lines'), label.size = NA,
             fill = NA) +
  scale_x_continuous('% valid', labels = ~ . * 100, breaks = seq(0, 1, 0.1)) +
  scale_y_continuous(expression(Delta~FCD)) +
  boxed_theme() +
  theme(aspect.ratio = 0.95)
p4
ggsave("fig/fig3/delta-FCD-vs-valid-scatterplot.pdf", p4, 
       width = 4.5, height = 4.5, units = "cm", useDingbats = FALSE)
