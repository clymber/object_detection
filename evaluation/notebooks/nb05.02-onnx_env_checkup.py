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
import json
import platform
from pathlib import Path

import onnx
import onnxruntime as ort

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
yolox_nano_sport_onnx: Path = (
    RUN_ROOT / 
    "sport_object"/
    "yolox_nano_20260922T211805"/
    "weights/best_ckpt.onnx"
)

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
print(json.dumps(available_providers, indent=4))

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
print(json.dumps(session.get_providers(), indent=4))

# %%
for input in session.get_inputs():
    print(f"INPUT  - name: {input.name}; shape: {input.shape}; type: {input.type}")
for output in session.get_outputs():
    print(f"OUTPUT - name: {output.name}; shape: {output.shape}; type: {output.type}")
