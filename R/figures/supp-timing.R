setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = F)
library(tidyverse)
library(magrittr)
library(broom)
source("R/theme.R")

###############################################################################-
## a. time to parse valid SMILES (SELFIES) ####
###############################################################################-

parse = readRDS("data/summary/timing/parse-valid.rds")
df1 = parse %>% 
  extract(, map_lgl(., ~ n_distinct(.) > 1)) %>% 
  mutate(representation = ifelse(!is.na(constraints),
                                 fct_recode(constraints, 
                                            'Unconstrained SELFIES' = 'none',
                                            'Texas SELFIES' = 'c5') %>% 
                                   as.character(),
                                 representation),
         time = time / (60*60))

labs1 = df1 %>% 
  group_by(representation) %>% 
  summarise(mean = mean(time), median = median(time)) %>% 
  ungroup() %>% 
  mutate(label = format(mean, format = 'f', digits = 2) %>% paste('h'))
p1 = df1 %>% 
  ggplot(aes(x = representation, y = time, color = representation, 
             fill = representation)) +
  geom_label(data = labs1, aes(y = -Inf, label = label), 
             size = 1.75, vjust = 0, label.padding = unit(0.6, 'lines'),
             fill = NA, label.size = NA) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_x_discrete(labels = ~ chartr(' ', '\n', .)) +
  scale_y_continuous('Time, hours', breaks = pretty_breaks(),
                     expand = expansion(c(0.17, 0.05))) + 
  scale_fill_manual(values = representation_pal) +
  scale_color_manual(values = representation_pal) +
  boxed_theme() +
  theme(aspect.ratio = 1.5, 
        legend.position = 'none',
        axis.text.x = element_text(angle = 45, hjust = 1, lineheight = 0.9),
        axis.title.x = element_blank())
p1
ggsave("fig/supp-timing/parse-valid-outputs.pdf", p1, 
       width = 5, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## b. schematic ####
###############################################################################-

###############################################################################-
## c. time to train models ####
###############################################################################-

train = readRDS("data/summary/timing/train-models.rds")
df2 = train  %>% 
  extract(, map_lgl(., ~ n_distinct(.) > 1)) %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES'),
         time = time / (60*60)) %>% 
  filter(stage == 'train')

labs2 = df2 %>% 
  group_by(representation) %>% 
  summarise(mean = mean(time), median = median(time)) %>% 
  ungroup() %>% 
  mutate(label = format(mean, format = 'f', digits = 2) %>% paste('h'))
p2 = df2 %>% 
  ggplot(aes(x = representation, y = time, color = representation, 
             fill = representation)) +
  geom_label(data = labs2, aes(y = -Inf, label = label), 
             size = 1.75, vjust = 0, label.padding = unit(0.6, 'lines'),
             fill = NA, label.size = NA) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_x_discrete(labels = ~ chartr(' ', '\n', .)) +
  scale_y_continuous('Time, hours', breaks = pretty_breaks(),
                     expand = expansion(c(0.17, 0.05))) + 
  scale_fill_manual(values = representation_pal) +
  scale_color_manual(values = representation_pal) +
  boxed_theme() +
  theme(aspect.ratio = 1.7, 
        legend.position = 'none',
        axis.text.x = element_text(angle = 45, hjust = 1, lineheight = 0.9),
        axis.title.x = element_blank())
p2
ggsave("fig/supp-timing/train-models.pdf", p2, width = 5, height = 5,
       units = "cm", useDingbats = FALSE)

###############################################################################-
## d. total time ####
###############################################################################-

## training
train = readRDS("data/summary/timing/train-models.rds")
train0 = train %>% 
  extract(, map_lgl(., ~ n_distinct(.) > 1))

## sampling
sample = readRDS("data/summary/timing/sample-models.rds")
sample0 = sample %>% 
  extract(, map_lgl(., ~ n_distinct(.) > 1))

## parsing
parse = readRDS("data/summary/timing/parse-valid.rds")
parse0 = parse %>% 
  extract(, map_lgl(., ~ n_distinct(.) > 1))

# combine
df3 = bind_rows(filter(train0, stage == 'train'),
                filter(train0, stage == 'sample'),
                filter(parse0, is.na(constraints)) %>% 
                  mutate(stage = 'parse',
                         # scale: 500k vs. 10m
                         time = time * 5e5 / 10e6)) %>% 
  group_by(representation, sample_idx) %>% 
  summarise(time = sum(time)) %>% 
  ungroup() %>% 
  mutate(time = time / (60*60),
         representation = fct_relevel(representation, 'SMILES', 'SELFIES'))

# plot
labs3 = df3 %>% 
  group_by(representation) %>% 
  summarise(mean = mean(time), median = median(time)) %>% 
  ungroup() %>% 
  mutate(label = format(mean, format = 'f', digits = 2) %>% paste('h'))
p3 = df3 %>% 
  ggplot(aes(x = representation, y = time, color = representation, 
             fill = representation)) +
  geom_label(data = labs3, aes(y = -Inf, label = label), 
             size = 1.75, vjust = 0, label.padding = unit(0.6, 'lines'),
             fill = NA, label.size = NA) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_x_discrete(labels = ~ chartr(' ', '\n', .)) +
  scale_y_continuous('Time, hours', breaks = pretty_breaks(),
                     expand = expansion(c(0.17, 0.05))) + 
  scale_fill_manual(values = representation_pal) +
  scale_color_manual(values = representation_pal) +
  boxed_theme() +
  theme(aspect.ratio = 1.7, 
        legend.position = 'none',
        axis.text.x = element_text(angle = 45, hjust = 1, lineheight = 0.9),
        axis.title.x = element_blank())
p3
ggsave("fig/supp-timing/total-time.pdf", p3, width = 5, height = 5,
       units = "cm", useDingbats = FALSE)

###############################################################################-
## e. total time, sample size=2.5m ####
###############################################################################-

# combine
df4 = bind_rows(filter(train0, stage == 'train'),
                filter(train0, stage == 'sample'),
                filter(parse0, is.na(constraints)) %>% 
                  mutate(stage = 'parse',
                         # scale: 500k vs. 10m
                         time = time * 4e6 / 10e6)) %>% 
  group_by(representation, sample_idx) %>% 
  summarise(time = sum(time)) %>% 
  ungroup() %>% 
  mutate(time = time / (60*60),
         representation = fct_relevel(representation, 'SMILES', 'SELFIES'))

# plot
labs4 = df4 %>% 
  group_by(representation) %>% 
  summarise(mean = mean(time), median = median(time)) %>% 
  ungroup() %>% 
  mutate(label = format(mean, format = 'f', digits = 2) %>% paste('h'))
p4 = df4 %>% 
  ggplot(aes(x = representation, y = time, color = representation, 
             fill = representation)) +
  geom_label(data = labs4, aes(y = -Inf, label = label), 
             size = 1.75, vjust = 0, label.padding = unit(0.6, 'lines'),
             fill = NA, label.size = NA) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_x_discrete(labels = ~ chartr(' ', '\n', .)) +
  scale_y_continuous('Time, hours', breaks = pretty_breaks(),
                     expand = expansion(c(0.17, 0.05))) + 
  scale_fill_manual(values = representation_pal) +
  scale_color_manual(values = representation_pal) +
  boxed_theme() +
  theme(aspect.ratio = 1.7, 
        legend.position = 'none',
        axis.text.x = element_text(angle = 45, hjust = 1, lineheight = 0.9),
        axis.title.x = element_blank())
p4
ggsave("fig/supp-timing/total-time-4m.pdf", p4, width = 5, height = 5,
       units = "cm", useDingbats = FALSE)
