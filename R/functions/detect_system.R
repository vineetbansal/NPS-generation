# detect system
system = 'sockeye'
node = Sys.info()[["nodename"]]
if (grepl('argo', node)) {
  system = 'argo'
} else if (Sys.info()["sysname"] == 'Darwin') {
  system = 'local'
}
# set up base directory
if (system == 'argo') {
  base_dir = '/Genomics/skinniderlab/invalid-smiles-analysis'
  args$allocation = 'ms0270'
} else if (system == 'sockeye') {
  base_dir = "/scratch/st-ljfoster-1/invalid-smiles-analysis"
  args$allocation = 'st-ljfoster-1'
} else if (system == 'local') {
  base_dir = "~/git/invalid-smiles-analysis/data"
  args$allocation = NULL
}
