setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = F)
library(tidyverse)
library(magrittr)
library(data.table)
library(broom)
source("R/theme.R")

# read an example dataset
dat = fread("data/summary/sample-molecules/sample-1-valid-1.csv.gz") %>% 
  set_colnames(c('loss', 'smiles', 'tokens', 'msg', 'error')) %>% 
  # tag valid vs. not
  mutate(valid = msg == 'valid') %>% 
  # filter empty samples
  filter(tokens > 0)

# read effect sizes
eff = readRDS("data/summary/sample-molecules/parse-valid-SMILES-RNN.rds") 

# read frequencies
freqs = eff$freqs %>% 
  replace_na(list(error = 'valid')) %>% 
  mutate(error = str_to_sentence(error))

###############################################################################-
## a. loss, valid vs. invalid, boxplot of a single model ####
###############################################################################-

# plot
range = boxplot(loss ~ valid, data = dat)$stats %>% range
p1 = dat %>% 
  ggplot(aes(x = valid, y = loss, color = valid, fill = valid)) +
  geom_violin(alpha = 0.15, color = NA, width = 0.7) +
  geom_boxplot(outlier.shape = NA, width = 0.6, alpha = 0.4, size = 0.35) + 
  scale_y_continuous('Loss', expand = c(0.1, 0)) +
  scale_x_discrete(labels = c('FALSE' = 'Invalid\nSMILES', 
                              'TRUE' = 'Valid\nSMILES')) +
  scale_fill_manual(values = unname(valid_pal)) +
  scale_color_manual(values = unname(valid_pal)) +
  coord_cartesian(ylim = range) +
  boxed_theme() +
  theme(aspect.ratio = 1.7,
        legend.position = 'none',
        axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1))
p1
ggsave("fig/fig2/example-loss-boxplot.pdf", p1,
       width = 5, height = 4, units = "cm", useDingbats = FALSE)

x = filter(dat, valid) %>% pull(loss)
y = filter(dat, !valid) %>% pull(loss)
wilcox.test(x, y)

###############################################################################-
## b. loss, valid vs. invalid, effect sizes ####
###############################################################################-

df2 = eff$effects %>% 
  filter(analysis == 'loss') %>% 
  filter(database == 'ChEMBL',
         representation == 'SMILES',
         n_molecules == 1e5,
         enum_factor == 0,
         min_tc == 0)
pal1 = grafify::graf_palettes$bright %>% tail(1) %>% unname()
p2 = df2 %>% 
  mutate(xval = 'Valid vs.\ninvalid\nSMILES') %>% 
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
p2
ggsave("fig/fig2/loss-d-boxplot.pdf", p2,
       width = 4, height = 4, units = "cm", useDingbats = FALSE)

# t-test
test = df2 %>% 
  do(tidy(t.test(.$d)))
message('loss, default settings: p = ', 
        format(test$p.value, digits = 2))

###############################################################################-
## c. loss, by error type, boxplot of a single model ####
###############################################################################-

range = boxplot(loss ~ error, data = dat)$stats %>% range
hline = median(dat$loss[dat$valid])
p3 = dat %>% 
  mutate(facet = ifelse(valid, 'Valid', 'Invalid'),
         error = str_to_sentence(error) %>% 
           replace(. == "", "Valid")) %>% 
  ggplot(aes(x = error, y = loss, fill = error, color = error)) +
  facet_grid(~ facet, scales = 'free_x', space = 'free') +
  geom_hline(aes(yintercept = hline), size = 0.45, color = 'grey88') +
  geom_violin(alpha = 0.15, color = NA, width = 0.7) +
  geom_boxplot(outlier.shape = NA, width = 0.6, alpha = 0.4, size = 0.35) + 
  scale_x_discrete('Error', labels = trimws) +
  scale_y_continuous('Loss', expand = c(0.1, 0)) +
  scale_fill_manual(values = error_type_pal) +
  scale_color_manual(values = error_type_pal) +
  coord_cartesian(ylim = range) +
  boxed_theme() +
  theme(legend.position = 'none',
        strip.text = element_blank(),
        axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1))
p3
ggsave("fig/fig2/example-loss-type-boxplot.pdf", p3,
       width = 5, height = 5, units = "cm", useDingbats = FALSE)

error_types = unique(dat$error) %>% setdiff("")
tests = map_dfr(error_types, ~ {
  dat0 = filter(dat, error %in% c("", .x)) 
  x = filter(dat0, valid) %>% pull(loss)
  y = filter(dat0, !valid) %>% pull(loss)
  wilcox.test(x, y) %>% tidy()
})
tests

###############################################################################-
## d. loss, by error type, effect sizes ####
###############################################################################-

# plot default
df4 = eff$effects %>% 
  filter(grepl(paste0("loss: "), analysis)) %>% 
  mutate(xval = gsub("^.*: ", "", analysis) %>% str_to_sentence()) %>% 
  filter(database == 'ChEMBL',
         representation == 'SMILES',
         n_molecules == 1e5,
         enum_factor == 0,
         min_tc == 0)
p4 = df4 %>% 
  ggplot(aes(x = xval, y = d, fill = xval, color = xval)) +
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
        # axis.ticks.x = element_blank(),
        legend.position = 'none',
        aspect.ratio = 1)
p4
ggsave("fig/fig2/loss-type-d-default.pdf", p4,
       width = 4, height = 4, units = "cm", useDingbats = FALSE)

# t-test
for (xval in unique(df4$xval)) {
  test = df4 %>% 
    filter(xval == !!xval) %>% 
    do(tidy(t.test(.$d)))
  message('loss, ', xval, ': p = ', format(test$p.value, digits = 2))
}

###############################################################################-
## e. loss, by error type, proportions ####
###############################################################################-

# plot default
df5 = freqs %>% 
  group_by_at(vars(-sample_idx, -proportion, -n)) %>% 
  summarise(mean = mean(proportion), sd = sd(proportion)) %>% 
  ungroup() %>% 
  filter(database == 'ChEMBL',
         representation == 'SMILES',
         n_molecules == 1e5,
         enum_factor == 0,
         min_tc == 0)
pal = grafify::graf_palettes$bright %>% unname()
p5 = df5 %>% 
  filter(error != 'Valid') %>% 
  ggplot(aes(x = '1', y = mean, fill = error, color = error)) +
  geom_col(width = 0.8, color = 'grey10', size = 0.15) + 
  scale_x_discrete(expand = c(0, 0)) +
  scale_y_continuous("% of SMILES", labels = ~ . * 100, expand = c(0, 0),
                     breaks = seq(0, 10, 2) / 100) +
  scale_color_manual('', values = error_type_pal) +
  scale_fill_manual('', values = error_type_pal) +
  # coord_polar(theta = 'y') + 
  boxed_theme() +
  theme(# axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        # axis.text.y = element_blank(),
        axis.text.x = element_blank(),
        # axis.ticks.y = element_blank(),
        axis.ticks.x = element_blank(),
        legend.position = 'right',
        legend.key.size = unit(0.35, 'lines'),
        # panel.border = element_blank(),
        aspect.ratio = 8)
p5
ggsave("fig/fig2/error-proportions-default-pie-chart.pdf", p5,
       width = 4.15, height = 4.15, units = "cm", useDingbats = FALSE)

###############################################################################-
## f. percent valid, by loss decile ####
###############################################################################-

# read likelihood by decile summary
decile = readRDS("data/summary/sample-molecules/valid-by-decile.rds") %>% 
  filter(database == 'ChEMBL', 
         min_tc == 0, 
         representation == 'SMILES', 
         n_molecules == 1e5, 
         enum_factor == 0)

# plot as boxplot
pal = brewer.pal(10, 'RdGy') %>% rev()
p6 = decile %>% 
  ggplot(aes(x = factor(decile), y = valid, fill = factor(decile), 
             color = factor(decile))) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_x_discrete('Loss, decile') +
  scale_y_continuous("% valid SMILES", labels = ~ . * 100) +
  scale_color_manual(values = pal) +
  scale_fill_manual(values = pal) +
  boxed_theme() +
  theme(legend.position = 'none',
        aspect.ratio = 0.9)
p6
ggsave("fig/fig2/pct-valid-by-loss-decile-boxplot.pdf", p6,
       width = 7, height = 3.75, units = "cm", useDingbats = FALSE)

# test
library(clinfun)
jonckheere.test(decile$decile, decile$valid)
