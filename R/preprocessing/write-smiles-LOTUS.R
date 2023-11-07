# Write a line-delimited list of SMILES from LOTUS.
setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = F)
library(tidyverse)
library(magrittr)

# read data frame
df = read_tsv("data/prior/LOTUS/LOTUS_DB.smi.gz", col_names = FALSE)

# extract SMILES
smiles = df[[1]] %>% 
  unique()

# write to txt
writeLines(smiles, "data/prior/raw/LOTUS.txt")
