setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = FALSE)
library(tidyverse)
library(magrittr)
library(broom)
source("R/theme.R")

###############################################################################-
## Read data: valid vs. invalid SELFIES ####
###############################################################################-

# read entire outcome distributions
## note: these are all for default parameters, sample_idx=1
train = read_csv("data/summary/outcome-distrs/outcome-distrs-train.csv.gz")
inv = read_csv("data/summary/outcome-distrs/outcome-distrs-constraints=none-invalid.csv.gz")
val = read_csv("data/summary/outcome-distrs/outcome-distrs-constraints=none-valid.csv.gz")

# combine
dat = bind_rows(
  train %>% mutate(xval = 'Training set'),
  inv %>% mutate(xval = 'Invalid'),
  val %>% mutate(xval = 'Valid')
) %>% 
  mutate(xval = fct_relevel(xval, 'Training set', 'Invalid', 'Valid'))

# read effect sizes
effs = readRDS("data/summary/outcome-distrs/effect-sizes-break-SELFIES.rds")
# filter to default params 
effs0 = map(effs, ~ filter(.x, constraints == 'none') %>% 
              mutate(subset = str_to_title(subset) %>%
                       fct_relevel('Invalid', 'Valid')))

###############################################################################-
## Effect sizes: valid vs. invalid SELFIES ####
###############################################################################-

# define discrete and continuous outcomes
continuous_outcomes = unique(effs0$continuous$outcome)
discrete_outcomes = c(
  "# of rings",
  "# of aliphatic rings",
  "# of aromatic rings",
  "# of hydrogen donors",
  "# of hydrogen acceptors"
)

# iterate through outcomes
pvals = data.frame()
eff_plots = list()
distr_plots = list()
for (outcome in setdiff(continuous_outcomes, 
                        c('# of aromatic rings',
                          '# of aliphatic rings',
                          'QED',
                          'Synthetic accessibility score',
                          'Natural product-likeness score'))) {
  outcome_clean = fct_recode(outcome,
                             'Topological complexity' = 'BertzTC',
                             'Topological polar surface area' = 'TPSA',
                             '% stereocenters' = '% stereocentres') %>% 
    as.character()
  
  #############################################################################-
  ## * distribution (Sankey/boxplot), SMILES vs. SELFIES ####
  #############################################################################-
  
  dat0 = filter(dat, outcome == !!outcome)
  
  if (outcome %in% discrete_outcomes) {
    ## Sankey plot
    dat0 %<>% 
      # winsorize
      mutate(value = ifelse(value >= 10, 10, value)) %>% 
      # cut
      mutate(value_c = cut(value, breaks = seq(-1, 11, 2)) %>% 
               as.character() %>% 
               gsub("\\(|\\]", "", .) %>% 
               fct_recode('0-1' = '-1,1',
                          '2-3' = '1,3',
                          '4-5' = '3,5',
                          '6-7' = '5,7',
                          '8-9' = '7,9',
                          '10+' = '9,11') %>% 
               fct_relevel('0-1', '2-3', '4-5', '6-7', '8-9', '10+'))
    
    # convert to proportions
    counts = dplyr::count(dat0, xval, value_c) %>% 
      # convert to proportions
      group_by(xval) %>% 
      mutate(prop = n / sum(n)) %>% 
      ungroup() %>% 
      mutate(value = fct_rev(value_c))
    
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
      scale_fill_manual(values = ring_pal, name = outcome) +
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
  } else {
    ## boxplot
    pal = valid_pal %>% 
      c('Training set' = 'grey82')
    boxplot_stats = boxplot(value ~ xval, data = dat0)$stats
    range = range(boxplot_stats)
    yint = boxplot_stats[3, 1]
    yint_df = head(dat0, 1) %>%
      cbind(yint = yint,
            x = c(-Inf, 1.3, 2.3, 3.3),
            xend = c(0.7, 1.7, 2.7, Inf))
    p1 = dat0 %>% 
      ggplot(aes(x = xval, y = value, fill = xval, color = xval)) +
      geom_blank() + 
      geom_segment(data = yint_df, aes(y = yint, yend = yint,
                                       x = x, xend = xend), 
                   color = 'grey88', size = 0.7) +
      geom_violin(alpha = 0.15, color = NA, width = 0.7) +
      geom_boxplot(outlier.shape = NA, width = 0.6, alpha = 0.4, size = 0.35) + 
      scale_x_discrete(labels = ~ fct_recode(., 'Valid\nSELFIES' = 'Valid',
                                             'Invalid\nSELFIES' = 'Invalid')) +
      scale_y_continuous(outcome_clean, expand = c(0.1, 0)) +
      scale_fill_manual(values = pal) +
      scale_color_manual(values = pal) +
      coord_cartesian(ylim = range) +
      boxed_theme() +
      theme(aspect.ratio = 1.4,
            legend.position = 'none',
            axis.title.x = element_blank(),
            axis.text.x = element_text(angle = 45, hjust = 1))
    p1
  }
  distr_plots[[outcome_clean]] = p1
  
  #############################################################################-
  ## * effect sizes (boxplot), SMILES vs. SELFIES ####
  #############################################################################-
  
  eff = effs0$continuous %>% filter(outcome == !!outcome)
  
  # test p-values
  pval = eff %>% 
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
  pvals %<>% bind_rows(mutate(pval, outcome = outcome_clean) %>% 
                         dplyr::select(outcome, everything()))
  
  # boxplot
  p2 = eff %>% 
    ungroup() %>% 
    distinct(subset, sample_idx, eff) %>% 
    mutate(outcome = outcome_clean) %>% 
    ggplot(aes(x = subset, y = -eff, color = subset, fill = subset)) +
    facet_wrap(~ outcome) +
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
  p2
  eff_plots[[outcome_clean]] = p2
}

pvals

## combine
row1 = distr_plots[[1]] + eff_plots[[1]] +
  distr_plots[[2]] + eff_plots[[2]] +
  distr_plots[[3]] + eff_plots[[3]] + 
  plot_layout(nrow = 1) &
  theme(plot.background = element_blank())
ggsave("fig/supp-outcome-distrs-valid-vs-invalid-SELFIES/row1.pdf", row1, 
       width = 19, height = 5, units = "cm", useDingbats = FALSE)

row2 = distr_plots[[4]] + eff_plots[[4]] +
  distr_plots[[5]] + eff_plots[[5]] +
  distr_plots[[6]] + eff_plots[[6]] + 
  plot_layout(nrow = 1) &
  theme(plot.background = element_blank())
ggsave("fig/supp-outcome-distrs-valid-vs-invalid-SELFIES/row2.pdf", row2, 
       width = 19, height = 5, units = "cm", useDingbats = FALSE)

row3 = distr_plots[[7]] + eff_plots[[7]] +
  distr_plots[[8]] + eff_plots[[8]] +
  plot_layout(nrow = 1) &
  theme(plot.background = element_blank())
ggsave("fig/supp-outcome-distrs-valid-vs-invalid-SELFIES/row3.pdf", row3, 
       width = 19, height = 5, units = "cm", useDingbats = FALSE)

row4 = distr_plots[[9]] + eff_plots[[9]] + 
  distr_plots[[10]] + eff_plots[[10]] +
  plot_layout(nrow = 1) &
  theme(plot.background = element_blank())
ggsave("fig/supp-outcome-distrs-valid-vs-invalid-SELFIES/row4.pdf", row4, 
       width = 19, height = 5, units = "cm", useDingbats = FALSE)

###############################################################################-
## Aromatic/aliphatic rings, valid vs. invalid SELFIES ####
###############################################################################-

## * aromatic rings ####
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
p3
ggsave("fig/supp-outcome-distrs-valid-vs-invalid-SELFIES/aromatic-rings-example-sankey.pdf", p3,
       width = 7, height = 4.5, units = "cm", useDingbats = FALSE)

## * aliphatic rings ####
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
ring_pal = pals::ocean.dense(6)
ring_pal = pals::kovesi.diverging_bwr_40_95_c42(6) %>% rev
# ring_pal = pals::kovesi.diverging_rainbow_bgymr_45_85_c67(6)
p4 = polygons %>% 
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
p4
ggsave("fig/supp-outcome-distrs-valid-vs-invalid-SELFIES/aliphatic-rings-example-sankey.pdf", p4,
       width = 7, height = 4.5, units = "cm", useDingbats = FALSE)

###############################################################################-
## 'Meta-volcano' plot, valid vs. invalid SELFIES ####
###############################################################################-

continuous_outcomes = unique(effs0$continuous$outcome) %>% 
  extract(!grepl('QED|Synthetic|Natural', .))
volcano = map_dfr(continuous_outcomes, ~ {
  continuous_outcome = .x
  # get that outcome
  eff = effs0$continuous %>% filter(outcome == continuous_outcome)
  
  # test p-values
  pvals = eff %>% 
    ungroup() %>% 
    dplyr::select(database, subset, n_molecules, min_tc, sample_idx, 
                  eff) %>%
    distinct() %>% 
    group_by(database, n_molecules, min_tc, sample_idx) %>% 
    filter(n_distinct(subset) == 2) %>% 
    summarise(delta = -eff[subset == 'Valid'] - 
                -eff[subset == 'Invalid']) %>% 
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
                  segment.size = 0.15, max.overlaps = 999) +
  scale_x_continuous(expression(Delta[Valid~-~Invalid]), limits = range) +
  scale_y_continuous(expression(-log[10](P))) +
  boxed_theme() +
  theme(aspect.ratio = 1) +
  ggh4x::force_panelsizes(rows = unit(26.24, 'mm'),
                          cols = unit(26.24, 'mm'))
p5
ggsave("fig/supp-outcome-distrs-valid-vs-invalid-SELFIES/meta-volcano.pdf", p5, 
       width = 5, height = 5, units = "cm", useDingbats = FALSE)
