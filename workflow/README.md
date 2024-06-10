## Setting up the Environment
### Della
Make sure that `conda` is in your path by loading the appropriate module: 

```
module load anaconda3/2023.9
```
You can also load a different anaconda version from all the available ones. To see all available modules, run `module avail anaconda3`.

For the Della cluster, no further actions are required to install conda or python. You can go ahead and create a new conda environment as instructed [below](https://github.com/skinniderlab/CLM/edit/aa/update_readme/workflow/README.md#creating-a-conda-environment).

**Important:** To access the Della cluster, you must first obtain an account by submitting a request. Once approved, you can connect to the cluster using SSH. [Click Here for Detailed Instructions](https://researchcomputing.princeton.edu/systems/della#access)

### Argo
The default conda is recommended on argo systems. You do not need to load any modules or do anything special to use the conda command on argo. [Click here for detailed information regarding conda on Argo](https://lsidocs.princeton.edu/index.php/Conda).

### Personal Machine
Before you can use the `clm` command on `Argo` or your personal machine, ensure that your system meets the following prerequisites:

- Python 3.10 or later
- Pip package manager

We recommend [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) for environment management. 

### Creating a Conda Environment

1. Create a new conda environment named `clm` with Python version 3.10.
   ```
   conda create --name clm python=3.10 pip
   ```
2. Activate the environment.
   ```
   conda activate clm
   ``` 
   The command prompt will change to indicate the new conda environment by prepending `(clm)`.

3. When you are done with the development, deactivate the `clm` environment and return to `(base)` by the following command:
   ```
   conda deactivate
   ```
   
## Installation 
1. Clone the repository and enter it:
   ```
   git clone https://github.com/skinniderlab/CLM.git
   cd clm
   ```
2. In the activated environment, run the following command in your terminal to install the CLM package in editable mode with any optional dependenices
    ```
    pip install -e .[dev]
    ```
## Snakemake workflow

The included `Snakefile` provides a way to run the end-end workflow for the
`clm` repository.

The following "rulegraph" illustrates the unique steps in the workflow that would be run,
and their order of execution.

![Rulegraph](rulegraph.png "Rulegraph")

Note that the actual number of jobs will exceed the number of rules above, since many different
instances of the rules will be run for any given dataset, depending on the number of train/test folds,
the number of independent seeds against which to run the models etc.

The following dependency graph illustrates the particular instances of the steps that would be run, when
running the workflow for a single `enum_factor`, 3 `folds`, 2 `training seeds`, and 1 `sampling seed`.
(This is the starter configuration provided in the file `config_fast.yaml`).

![DAG](dag.png "DAG")


### Testing the workflow

To run this workflow on a tiny dataset provided with `clm`:

#### Steps

1. `cd` to the `workflow/` folder and run the following command to see the steps (including the actual commands) that will be run:

```
cd CLM/workflow
snakemake --configfile config/config_fast.yaml --jobs 1 --dry-run -p
```

2. Repeat the command without the `--dry-run -p` to execute the workflow. The end-end workflow should take around 5 minutes on computers where `torch` has access to a gpu, or 20-25 minutes otherwise.

Note that the configuration provided in `config_fast.yaml`, as well as the truncated datasets that it uses by default, are purely for software testing purposes, and generate results/graphs that are not interpretable. To run the actual workflow on your dataset, read on.

### Running the "real" workflow

To run the end-end workflow on an actual dataset (preferably on a cluster), repeat the above process, but with a few tweaks:

#### Steps

a. Specify the paths to your dataset (a `.txt` file containing SMILES in each line), as well as the [PubChem]() dataset. (link coming soon), as `--config` parameters on the command line.

b. Add the `--slurm` flag to indicate that the steps should be run using `sbatch`.

c. Replace `--configfile config/config_fast.yaml` with `--configfile config/config.yaml` (or eliminate this flag altogether).

d. Make any other configuration changes in `config.yaml` (network architecture, number of epochs, other hyperparameters).

e. Increase the value of the `--jobs` flag to specify the maximum number of slurm jobs to run at a time.

Run the following command to see the steps (including the actual commands) that will be run:

```
snakemake --config dataset=/path/to/dataset.txt pubchem_tsv_file=/path/to/PubChem.tsv --jobs 10 --dry-run -p
```

Replace `dataset` and `pubchem_tsv_file` with the paths to your dataset file and PubChem tsv file respectively. These will override the
values obtained for these flags in `config.yaml`. Alternately, you can change `config.yaml` to point to the correct paths.

Repeat the command without the `--dry-run -p` to execute the workflow. The end-end workflow should take around 24 hours, depending on the cluster workload and the exact configuration in `config.json`.

```
snakemake --config dataset=/path/to/dataset.txt pubchem_tsv_file=/path/to/PubChem.tsv --jobs 10 &
```

> Note that running `snakemake` in a foreground process will run the workflow in blocking mode. Though actual jobs will be submitted to compute nodes, pressing
Ctrl-C will cause `snakemake` to attempt to cancel pending/currently running jobs (through `scancel`). You should thus run the actual workflow in the background
using `&`, or use an environment like [tmux](https://github.com/tmux/tmux/wiki/Getting-Started) that you can attach to and detach from on demand.



### Other useful commands

To generate the Rule graph that you see above:
```
snakemake --configfile .. --forceall --rulegraph | dot -Tpng > rulegraph.png
```

To generate the DAG dependency graph that you see above:
```
snakemake --configfile .. --forceall --dag | dot -Tpng > dag.png
```
