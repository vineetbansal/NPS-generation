## Setting up the Environment

Before you can use the `clm` package, ensure that your system meets the following prerequisites:

- Python 3.10 or later
- Pip package manager

#### Creating a Conda Environment

`clm` can get all its dependencies from `pypi` using `pip` and does not need [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) for environment management.
Nevertheless, this might be the easiest option for most users.

1. Create a new conda environment named `clm` with Python version 3.10.
   ```
   conda create --name clm python=3.10 pip
   ```
2. Activate the environment.
   ```
   conda activate clm
   ```
   The command prompt will change to indicate the new conda environment by prepending `(clm)`.

## Installation
1. Clone the repository and enter it:
   ```
   git clone https://github.com/skinniderlab/CLM.git
   cd clm
   ```
2. In the activated environment, install the dependencies provided in `environment.yml`:
    ```
    conda env update --file environment.yml
    ```
3. In the activated environment, run the following command in your terminal to install the CLM package in editable mode.
    ```
    pip install -e .
    ```

### Installation on specific clusters

The following instructions apply to installing and using `clm` on specific clusters at
Princeton University where we have tested out the package in the past.

#### Della
Make sure that `conda` is in your path by loading the appropriate module:

```
module load anaconda3/2023.9
```
You can also load a different anaconda version. To see all available modules, run `module avail anaconda3`.

No further actions are required to install conda or python. You can go ahead and create a new conda environment as instructed above.

#### Argo
The default conda is recommended on argo systems. You do not need to load any modules or do anything special to use the conda command on argo. [Click here for detailed information regarding conda on Argo](https://lsidocs.princeton.edu/index.php/Conda).
