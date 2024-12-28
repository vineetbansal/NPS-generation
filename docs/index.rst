.. CLM documentation master file, created by
   sphinx-quickstart on Thu Apr 18 03:52:24 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CLM's documentation!
===============================

A package to train and evaluate language models of chemical structures,
as used in the manuscript, :cite:t:`Qiang2024`.

Note that training and evaluating chemical language models requires significant
computational resources. By default, this package trains and samples from
multiple language models using cross-validation, and executes a comprehensive
suite of assessments to evaluate the quality of the generated structures.
For this reason, access to a high-performance computing cluster with GPU
resources is strongly recommended.

If you are simply looking to use the trained DeepMet model to annotate
metabolomics data, consider using the
DeepMet `web application <https://deepmet.org/>`_.

.. toctree::
   :titlesonly:

   clm/installation.md
   clm/workflow.md
   clm/workflow_steps.md

.. bibliography::
