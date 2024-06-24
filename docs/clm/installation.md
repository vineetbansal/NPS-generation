# Installation

This package is tested on Linux/ Mac OS X. Pre-built binaries should be available 
for platform-specific dependencies. No manual compilation should be needed.

For end users who simply want to use or run scripts depending on CLM, installing the `clm` package from PyPi is sufficient. 

Note: 
Installing the package installs CLM to the `site-packages` folder of your active environment. This is only desirable if you are not going to be doing any 
development on CLM, and only intend to run scripts that depend on the CLM package. 

For those who wish to develop, we recommend starting with the instructions on our README (copied below). 
Although not explicitly required, for developers and users not confident in software management, the use of `conda` is strongly encouraged. 

## Install Conda 
To follow the suggested installation, you will have to install Conda for Python3, either [Anaconda](https://www.anaconda.com/download/#linux) 
or [Miniconda](https://conda.io/miniconda.html), click on the right distribution to view Conda's installation instructions. 

Note:
If you're not sure which distribution is right for you, go with Miniconda. 

## Getting Started - Installation

Below we pip install the `clm` package using the `-e` flag to install the project in editable mode. The `'.[dev]'` command
installs `clm` from the local path with additional development tools such as pytest. See the [pip documentation](https://pip.pypa.io/en/stable/cli/pip_install/#options) for more details 
on using pip install 

```
# Clone the code 
git clone 
cd CLM 

# Create a new conda environment 
conda create --name clm python=3.10 pip 

# Activate the conda environment 
conda activate clm 

# Install the `clm` package from the checked out code 
# with the additional `dev` extension 
pip install -e '.[dev]'

```

## Testing the Package
Once you're done with the installation, make sure you can run all the 
unit tests correctly by typing the command: 
```
pytest
```
Tests currently take around 15 minutes to run, but this depends on your specific machine's resources and configuration. 

Also see [Running the Snakemake Workflow](snakemake.md)
Also see [Frequently Asked Questions](faq.md)
