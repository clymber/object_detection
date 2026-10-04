# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags
#     notebook_metadata_filter: jupytext,title,authors,-kernelspec,-jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # ONNX Runtime Environment Checkup

# %%
"""
Evaluation on ONNX exports.
"""
# %load_ext autoreload
# %autoreload 2
# %aimport -onnx

# %%
import platform
import sys
from pathlib import Path

import onnx
import onnxruntime as ort
from detection_common.utils.json_io import write_json

from detection_evaluation import find_latest_path
from detection_evaluation.config import (
    # DATA_ROOT,
    OUTPUT_ROOT,
    # WORKSPACE_ROOT,
)

# %%
print(f"ONNX version: {onnx.__version__!r}")
print(f"Opset: {onnx.defs.onnx_opset_version()}")
print(f"IR version: {onnx.IR_VERSION}") # Intermediate representation version

# %% [markdown]
# ## Load and validate the model struture
#
# ONNX is a model representation, or an interchange format, not an inference engine.

# %%
RUN_ROOT = OUTPUT_ROOT / "runs"
onnx_path = find_latest_path(RUN_ROOT, "sport_object/yolox_nano*/**/*.onnx")
if onnx_path is None:
    raise FileNotFoundError("No YOLOX nano run contains a weights/*.onnx file")

yolox_nano_sport_onnx: Path = onnx_path
model = onnx.load(yolox_nano_sport_onnx)
try:
    onnx.checker.check_model(model)
except onnx.checker.ValidationError as err:
    print("Invalid ONNX model:", err)
    raise


# %% [markdown]
# ### Model graph

# %%
# print(model.graph)

# %% [markdown]
# ### Inspect opset
#
# An opset specifies which version of the operator definitions the model uses.

# %%
for opset in model.opset_import:
    print(opset.domain, opset.version)

# %% [markdown]
# ## Execution Providers
#
# ONNX Runtime delegates operations to an **Execution Provider (EP)**.
# ONNX Runtime is the set of resources that actually executes the model. It's the
# hardware/backend implementation.

# %%
available_providers = ort.get_available_providers()
print("Available providers:")
write_json(sys.stdout, available_providers)

# %%
system = platform.system() # `Linux`, `Windows` or `Darwin`

if system == "Linux" and "CUDAExecutionProvider" in available_providers:
    ort.preload_dlls(directory="")

session = ort.InferenceSession(
    yolox_nano_sport_onnx,
    providers=[
        "CUDAExecutionProvider",    # Try NVDIA CUDA first
        "CPUExecutionProvider",     # Fall back to CPU when needed
    ],
)

print("Registered execution providers:")
write_json(sys.stdout, session.get_providers())

# %%
for input in session.get_inputs():
    print(f"INPUT  - name: {input.name}; shape: {input.shape}; type: {input.type}")
for output in session.get_outputs():
    print(f"OUTPUT - name: {output.name}; shape: {output.shape}; type: {output.type}")
