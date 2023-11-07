# Write a line-delimited list of SMILES from NORMAN.
setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = F)
library(tidyverse)
library(magrittr)

# read data frame
df = read.csv("data/prior/NORMAN/NORMANSusDat_20Nov2019_wExpoHaz.csv.gz")

# extract SMILES
smiles = df$SMILES %>% 
  unique()

# write to txt
writeLines(smiles, "data/prior/raw/NORMAN.txt")
