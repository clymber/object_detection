"""
Paths and on-demand format conversion for a versioned COCO dataset.
"""

from pathlib import Path
from tempfile import TemporaryDirectory


class Dataset:
    """
    Keep COCO canonical and reuse YOLO exports for an immutable dataset version.
    """

    def __init__(self, root: str | Path) -> None:
        """
        Store the dataset root without creating files or importing converters.
        """
        self.root = Path(root).expanduser().resolve()

    @property
    def coco(self) -> Path:
        """
        Return the canonical COCO directory, requiring it to exist.
        """
        path = self.root / "coco"
        if not path.is_dir():
            raise FileNotFoundError(f"COCO dataset not found: {path}")
        return path

    @property
    def yolo(self) -> Path:
        """
        Return the YOLO directory, converting COCO once if it is absent.
        """
        path = self.root / "yolo"
        if path.exists() or path.is_symlink():
            if not (path / "data.yaml").is_file():
                raise ValueError(f"Incomplete YOLO export (missing data.yaml): {path}")
            return path

        coco = self.coco
        with TemporaryDirectory(prefix=".yolo-", dir=self.root) as temporary:
            staging = Path(temporary) / "yolo"
            self._export_yolo(coco, staging)
            if not (staging / "data.yaml").is_file():
                raise ValueError("YOLO conversion did not create data.yaml")
            staging.rename(path)
        return path

    @staticmethod
    def _export_yolo(coco: Path, destination: Path) -> None:
        """
        Convert COCO with Datumaro, keeping media hardlinked when possible.
        """
        import datumaro as dm

        from .datumaro import (
            ExportFormat,
            ImportFormat,
            prefer_hardlinked_datumaro_media,
        )

        dataset = dm.Dataset.import_from(str(coco), format=ImportFormat.COCO_INSTANCES)
        with prefer_hardlinked_datumaro_media():
            dataset.export(
                str(destination),
                format=ExportFormat.YOLO_ULTRALYTICS,
                save_media=True,
            )

