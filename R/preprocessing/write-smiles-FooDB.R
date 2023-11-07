# Write a line-delimited list of SMILES from FooDB.
setwd("~/git/invalid-smiles-analysis")
options(stringsAsFactors = F)
library(tidyverse)
library(magrittr)
library(jsonlite)

# read JSON
json = stream_in(file("data/prior/FooDB/Compound.json.gz"))

# extract SMILES
smiles = json$moldb_smiles %>% 
  unique()

# write to txt
writeLines(smiles, "data/prior/raw/FooDB.txt")
