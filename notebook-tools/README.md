# Notebook Tools

The `notebook-tools` Conda environment owns JupyterLab, Jupytext, and
nbconvert. It edits and synchronizes notebooks but does not install model
frameworks; each notebook executes in its own project kernel.

From the workspace root, run `bash scripts/setup_conda_envs.sh notebook-tools`
and `make sync-notebooks`. Choose the namespaced project kernel when opening a
notebook. Generated `.ipynb` files are ignored by Git; paired `py:percent`
sources are committed.

The setup script reads environment-macos-mps.yml on Apple Silicon or
environment-linux-cuda.yml on Renku. The Linux profile has no CUDA
framework packages. Bootstrap Renku Miniforge with
bash scripts/install_miniforge_renku.sh before setup.
