"""Train and evaluate an Anomalib PatchCore baseline on MVTec AD bottle."""

from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "mvtec"
CATEGORY = "bottle"


def main() -> None:
    """Train PatchCore on normal images and report image-level AUROC."""
    category_path = DATA_ROOT / CATEGORY
    if not category_path.is_dir():
        msg = (
            f"MVTec AD bottle data was not found at {category_path}. "
            "Download and extract MVTec AD manually so this folder exists."
        )
        raise FileNotFoundError(msg)

    # Load the bottle category with no worker processes for Windows compatibility.
    datamodule = MVTecAD(
        root=DATA_ROOT,
        category=CATEGORY,
        num_workers=0,
    )

    # PatchCore learns a memory bank from the normal training images.
    model = Patchcore()

    # Use an available accelerator automatically, with one device.
    engine = Engine(
        accelerator="auto",
        devices=1,
        default_root_dir=PROJECT_ROOT / "results",
    )

    # Fit the model, then evaluate it against the dataset's test split.
    engine.fit(model=model, datamodule=datamodule)
    test_results = engine.test(model=model, datamodule=datamodule)

    if not test_results:
        raise RuntimeError("Anomalib returned no test metrics.")

    # Anomalib versions may include a test prefix in the metric key.
    metrics = test_results[0]
    for metric_name in ("image_AUROC", "test_image_AUROC"):
        if metric_name in metrics:
            image_auroc = float(metrics[metric_name])
            print(f"Image AUROC: {image_auroc:.4f}")
            break
    else:
        available_metrics = ", ".join(metrics.keys())
        raise RuntimeError(
            "Image AUROC was not returned by Anomalib. "
            f"Available metrics: {available_metrics or 'none'}"
        )


if __name__ == "__main__":
    main()
