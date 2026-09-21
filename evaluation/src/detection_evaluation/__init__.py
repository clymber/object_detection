"""
Framework-neutral evaluation and versioned prediction artifacts.
"""

from .baseline import BaselineExport, export_baseline, latest_run_dir, prepare_baseline
from .comparison import (
    LOGICAL_DATASET,
    MODEL_NAMES,
    BundleSelection,
    discover_bundles,
    write_bundle_comparisons,
)
from .metrics import (
    benchmark_predict,
    evaluate_predictions,
    file_sha256,
    read_prediction_artifact,
    write_comparison,
    write_prediction_artifact,
)
from .producer import export_predictions
from .run_protocol import (
    BUNDLE_MANIFEST_FILE_NAME,
    RUN_PROTOCOL_FILE_NAME,
    TRAINING_FILE_NAME,
    capture_training_hardware,
    create_run_protocol,
    finalize_training_attempt,
    publish_bundle,
    read_bundle,
    read_run_protocol,
    read_training_record,
    start_training_attempt,
)

__all__ = [
    "BaselineExport",
    "BUNDLE_MANIFEST_FILE_NAME",
    "BundleSelection",
    "benchmark_predict",
    "capture_training_hardware",
    "create_run_protocol",
    "discover_bundles",
    "evaluate_predictions",
    "export_baseline",
    "export_predictions",
    "file_sha256",
    "finalize_training_attempt",
    "latest_run_dir",
    "LOGICAL_DATASET",
    "MODEL_NAMES",
    "prepare_baseline",
    "publish_bundle",
    "read_bundle",
    "read_prediction_artifact",
    "read_run_protocol",
    "read_training_record",
    "RUN_PROTOCOL_FILE_NAME",
    "start_training_attempt",
    "TRAINING_FILE_NAME",
    "write_comparison",
    "write_bundle_comparisons",
    "write_prediction_artifact",
]
