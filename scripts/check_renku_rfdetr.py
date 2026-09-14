"""Exercise RF-DETR Small on one synthetic CUDA image in the Renku profile."""

from __future__ import annotations

import os
from pathlib import Path

workspace_root = Path(__file__).resolve().parents[1]
cache_roots = {
    "RF_HOME": workspace_root / "models/pretrained/rfdetr",
    "HF_HOME": workspace_root / "models/cache/rfdetr/huggingface",
    "TORCH_HOME": workspace_root / "models/cache/rfdetr/torch",
    "XDG_CACHE_HOME": workspace_root / "models/cache/rfdetr/xdg",
    "MPLCONFIGDIR": workspace_root / "models/cache/rfdetr/matplotlib",
}
for key, path in cache_roots.items():
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(key, str(path))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from rfdetr import RFDETRSmall  # noqa: E402

if not torch.cuda.is_available():
    raise SystemExit("RF-DETR smoke check requires an allocated CUDA GPU")
model = RFDETRSmall(device="cuda", resolution=640, positional_encoding_size=40)
predictions = model.predict(Image.new("RGB", (640, 640)), threshold=0.99)
torch.cuda.synchronize()
if predictions.xyxy.ndim != 2 or predictions.xyxy.shape[1] != 4:
    raise SystemExit("RF-DETR returned unexpected prediction geometry")
print(f"RF-DETR Small synthetic prediction passed on {torch.cuda.get_device_name(0)}")
