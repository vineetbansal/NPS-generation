setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = F)
library(tidyverse)
library(magrittr)
library(broom)
source("R/theme.R")

###############################################################################-
## c. % valid, default settings ####
###############################################################################-

# read data
dat = readRDS("data/summary/calculate-outcomes/calculate-outcomes-RNN.rds") 

# extract percent valid molecules
valid = filter(dat, outcome == '% valid')

# plot default
df1 = valid %>% 
  filter(database == 'ChEMBL',
         n_molecules == 1e5,
         enum_factor == 0,
         min_tc == 0) %>% 
  mutate(representation = fct_relevel(representation, 'SMILES', 'SELFIES'))
stopifnot(nrow(df1) == 20)
p1 = df1 %>% 
  ggplot(aes(x = representation, y = value, fill = representation, color = representation)) +
  geom_boxplot(alpha = 0.4, width = 0.6, size = 0.35, outlier.shape = NA) + 
  geom_jitter(size = 0.9, width = 0.1, height = 0, shape = 21, color = 'black', 
              stroke = 0.2) +
  scale_y_continuous('% valid', labels = ~ . * 100,
                     breaks = seq(0.88, 1, 0.02)) +
  scale_color_manual(values = representation_pal) +
  scale_fill_manual(values = representation_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1.7)
p1
ggsave("fig/fig1/pct-valid-default.pdf", p1,
       width = 4, height = 4, units = "cm", useDingbats = FALSE)

# t-test
test = df1 %>% 
  group_by(sample_idx) %>% 
  summarise(delta = value[representation == 'SMILES'] -
           value[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  do(tidy(t.test(.$delta)))
message('% valid, default settings: p = ', format(test$p.value, digits = 2))

# mean, median
df1 %>% 
  group_by(representation) %>% 
  do(tidy(summary(.$value)))

###############################################################################-
## d. FCD, default settings ####
###############################################################################-

# extract FCD
fcd = filter(dat, grepl('Frechet', outcome))

# print grid
dplyr::count(fcd, database, representation, enum_factor, n_molecules, min_tc)

# plot default
df2 = fcd %>% 
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
  scale_y_continuous('Frechet ChemNet distance') +
  scale_color_manual(values = representation_pal) +
  scale_fill_manual(values = representation_pal) +
  boxed_theme() +
  theme(axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none',
        aspect.ratio = 1.7)
p2
ggsave("fig/fig1/FCD-default.pdf", p2,
       width = 4, height = 4, units = "cm", useDingbats = FALSE)

# t-test
test = df2 %>% 
  group_by(sample_idx) %>% 
  summarise(delta = value[representation == 'SMILES'] -
           value[representation == 'SELFIES']) %>% 
  ungroup() %>% 
  do(tidy(t.test(.$delta)))
message('FCD, default settings: p = ', format(test$p.value, digits = 2))

###############################################################################-
## e. Correlation, % valid vs. delta-FCD ####
###############################################################################-

# combine all
df3 = fcd %>% 
  # compute cohen's d
  group_by_at(vars(-representation, -value)) %>% 
  filter(n_distinct(representation) == 2) %>% 
  summarise(delta_fcd = value[representation == 'SMILES'] -
              value[representation == 'SELFIES']) %>% 
  ungroup()
# merge in valid
valid0 = filter(valid, representation == 'SMILES')
df3 %<>% left_join(valid0, by = c('database', 'enum_factor', 'n_molecules',
                                   'min_tc', 'sample_idx'))

# correlation as legend
library(broom)
cor = df3 %>% do(tidy(cor.test(.$value, .$delta_fcd))) %>% 
  mutate(label = paste0('r = ', format(estimate, digits = 2), 
                        '\np = ', format(p.value, digits = 2)))

# plot
p3 = df3 %>% 
  ggplot(aes(x = value, y = delta_fcd)) +
  geom_jitter(shape = 21, size = 0.9, stroke = 0.25, color = 'black',
             fill = 'grey88', height = 0.01, width = 0.01) +
  geom_smooth(method = 'lm', color = 'red', size = 0.25, alpha = 0.15) +
  geom_label(data = cor, aes(x = -Inf, y = Inf, label = label), 
             hjust = 0, vjust = 1, size = 1.75, lineheight = 0.95,
             label.padding = unit(0.7, 'lines'), label.size = NA,
             fill = NA) +
  scale_x_continuous('% valid', labels = ~ . * 100, limits = c(NA, 1)) +
  scale_y_continuous(expression(Delta~FCD[SMILES~-~SELFIES])) +
  boxed_theme() +
  theme(aspect.ratio = 1)
p3
ggsave("fig/fig1/FCD-vs-valid-scatterplot.pdf", p3, 
       width = 4.5, height = 4.5, units = "cm", useDingbats = FALSE)
