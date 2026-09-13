# Notebook Tools

The `notebook-tools` Conda environment owns JupyterLab, Jupytext, and
nbconvert. It edits and synchronizes notebooks but does not install model
frameworks; each notebook executes in its own project kernel.

From the workspace root, run `bash scripts/setup_conda_envs.sh notebook-tools`
and `make sync-notebooks`. Choose the namespaced project kernel when opening a
notebook. Generated `.ipynb` files are ignored by Git; paired `py:percent`
sources are committed.
