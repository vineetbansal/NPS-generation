setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = F)
library(tidyverse)
library(magrittr)
library(broom)
source("R/theme.R")
library(pammtools) ## geom_stepribbon

# read data
ever_gen = readRDS("data/summary/prior/ever-generated.rds") %>% 
  filter(representation == 'SMILES')
topk = readRDS("data/summary/prior/top-k-accuracy.rds") %>% 
  filter(representation == 'SMILES')
tc = readRDS("data/summary/prior/Tc.rds") %>% 
  filter(representation == 'SMILES')
cands = readRDS("data/summary/prior/num-candidates.rds") %>% 
  filter(representation == 'SMILES')

# average over CV folds
topk0 = topk %>% 
  filter(!grepl('^f', cv_fold)) %>% 
  group_by_at(vars(-cv_fold, -n, -top_k)) %>% 
  summarise(mean = mean(top_k),
            sd = sd(top_k)) %>% 
  ungroup() %>% 
  extract(, map_lgl(., ~ n_distinct(.x) > 1))

# read pairwise Tc
pairwise = read_csv("data/summary/prior/pairwise_Tc/LOTUS-pairwise-Tc.csv.gz")
# create function that maps Tc to quantile
tc_quantile = function(tc) {
  cdf = ecdf(pairwise$tc)
  cdf(tc)
}

###############################################################################-
## a. Tc histogram, LOTUS ####
###############################################################################-

model_tc = readRDS("data/summary/prior/Tc-all-LOTUS.rds") %>% 
  filter(representation == 'SMILES',
         target_source == 'model') %>% 
  do(tidy(summary(.$Tc)))
print(model_tc)

df1 = pairwise

# plot mean Tc in a random fold against histogram of pairwise Tc
xintercept = model_tc$mean
label = round(100 * tc_quantile(xintercept), digits = 1) %>% 
  paste0('% of pairs') %>% 
  paste0('Mean Tc = ', format(xintercept, digits = 2), '\n≥', .)
p1 = df1 %>% 
  ggplot() +
  geom_histogram(aes(x = tc), size = 0.2, fill = 'grey96', color = 'grey82') +
  geom_vline(data = data.frame(xintercept = xintercept),
             aes(xintercept = xintercept), linetype = 'dotted', size = 0.15) +
  geom_label(data = data.frame(label = label),
             aes(x = xintercept, y = Inf, label = label), hjust = 0, vjust = 1,
             label.padding = unit(0.6, 'lines'), label.size = NA,
             size = 1.75, fill = NA, lineheight = 0.95) +
  scale_x_continuous('Tanimoto coefficient') +
  scale_y_continuous('Density', expand = expansion(c(0, 0.1))) +
  boxed_theme() +
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        aspect.ratio = 0.8,
        plot.title = element_text(size = 5))
p1
ggsave("fig/supp-prior/pairwise-Tc-histogram-LOTUS.pdf", p1, 
       width = 6, height = 4, units = "cm", useDingbats = FALSE)

###############################################################################-
## b. top-1/top-k, COCONUT, vs. PubChem ####
###############################################################################-

df2 = filter(topk0, database == 'COCONUT', 
             target_source != 'train') %>% 
  mutate(target_source = fct_recode(target_source,
                                    'Language model' = 'model') %>% 
           fct_relevel('Language model', 'PubChem'))

p2a = df2 %>% 
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
  coord_cartesian(ylim = c(0, 0.058)) +
  boxed_theme() +
  theme(aspect.ratio = 2.2,
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.x = element_blank(),
        legend.position = 'none')
p2a

p2b = df2 %>% 
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
p2b

p2 = p2a | p2b
p2
ggsave("fig/supp-prior/accuracy-COCONUT-vs-PubChem.pdf", p2, 
       width = 10, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## c. top-1/top-k, formula, COCONUT, vs. PubChem ####
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
df3 = filter(topk0_f, database == 'COCONUT', representation == 'SMILES',
            target_source != 'train'
             ) %>% 
  mutate(target_source = fct_recode(target_source,
                                    'Training set' = 'train',
                                    'Language model' = 'model') %>% 
           fct_relevel('Language model', 'Training set', 'PubChem'))

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
ggsave("fig/supp-prior/accuracy-COCONUT-vs-PubChem-formula.pdf", p3, 
       width = 10, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## d. top-1/top-k, formula, FooDB, vs. PubChem ####
###############################################################################-

# extract data for plotting
df4 = filter(topk0_f, database == 'FooDB', representation == 'SMILES',
             target_source != 'train') %>% 
  mutate(target_source = fct_recode(target_source,
                                    'Training set' = 'train',
                                    'Language model' = 'model') %>% 
           fct_relevel('Language model', 'Training set', 'PubChem'))

# plot top-1
p4a = df4 %>% 
  filter(rank == 1) %>% 
  ggplot(aes(x = target_source, y = mean, 
             fill = target_source, color = target_source)) +
  geom_col(aes(y = Inf), fill = 'grey96', color = NA, width = 0.8) +
  geom_col(aes(y = mean), width = 0.8) +
  geom_errorbar(aes(ymin = mean, ymax = mean + sd), size = 0.35, width = 0) +
  geom_label(aes(y = -Inf, label = round(100 * mean, digits = 1) %>% paste0('%')), 
             size = 1.75, vjust = 0, label.padding = unit(0.6, 'lines'),
             fill = NA, label.size = NA, color = 'white', alpha = 0.7) +
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
p4a

# plot top-k
p4b = df4 %>% 
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
p4b

p4 = p4a | p4b
p4
ggsave("fig/supp-prior/accuracy-FooDB-vs-PubChem-formula.pdf", p4, 
       width = 10, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## e. top-1/top-k, formula, NORMAN, vs. PubChem ####
###############################################################################-

# extract data for plotting
df5 = filter(topk0_f, database == 'NORMAN', representation == 'SMILES',
              target_source != 'train') %>% 
  mutate(target_source = fct_recode(target_source,
                                    'Training set' = 'train',
                                    'Language model' = 'model') %>% 
           fct_relevel('Language model', 'Training set', 'PubChem'))

# plot top-1
p5a = df5 %>% 
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
  coord_cartesian(ylim = c(0, 0.83)) +
  boxed_theme() +
  theme(aspect.ratio = 2.2,
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.x = element_blank(),
        legend.position = 'none')
p5a

# plot top-k
p5b = df5 %>% 
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
p5b

p5 = p5a | p5b
p5
ggsave("fig/supp-prior/accuracy-NORMAN-vs-PubChem-formula.pdf", p5, 
       width = 10, height = 5, units = "cm", useDingbats = FALSE)
