# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags
#     formats: ipynb,py:percent
#     notebook_metadata_filter: jupytext,title,authors,-kernelspec,-jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Basketball runtime comparison
#
# Recompute accuracy from saved full-split artifacts, without loading model
# frameworks or running inference. Failed or incomplete export results remain visible.
# Use the object-detection-evaluation kernel. Host timing groups must match
# before comparing latency. They do not measure iPhone performance.

# %%
from IPython.display import display
from detection_common import configure_stdio_relative_path
from detection_evaluation.config import DATA_ROOT, OUTPUT_ROOT, WORKSPACE_ROOT
from detection_evaluation.runtime_comparison import compare_runtime_results

configure_stdio_relative_path(WORKSPACE_ROOT)

# %% [markdown]
# ## Selections
#
# Empty mappings select the latest completed run, then its saved export result.
# RUN_SELECTIONS maps model names to exact run paths. EXPORT_SELECTIONS maps model
# names to exact model_manifest.json paths and derives their source run. Both
# overrides must agree when supplied together. Invalid overrides raise.

# %%
RUNS_DIR = OUTPUT_ROOT / "runs" / "basketball"
RUN_SELECTIONS = {}
EXPORT_SELECTIONS = {}
DATASET_DIR = DATA_ROOT / "composed" / "basketball-v0.1" / "coco"
COMPARISON_DIR = OUTPUT_ROOT / "comparisons" / "basketball_runtime"

# %%
comparison = compare_runtime_results(
    runs_dir=RUNS_DIR, dataset_dir=DATASET_DIR,
    output_dir=COMPARISON_DIR,
    run_selections=RUN_SELECTIONS, export_selections=EXPORT_SELECTIONS,
)
display(comparison["table"])
print(f"CSV: {comparison['csv_path']}")
print(f"JSON: {comparison['json_path']}")
print(f"Markdown: {comparison['markdown_path']}")

# %% [markdown]
# accepted=False retains failed, incomplete or parity-failed configurations as
# diagnostic evidence. Unavailable measurements have no invented ranking.
