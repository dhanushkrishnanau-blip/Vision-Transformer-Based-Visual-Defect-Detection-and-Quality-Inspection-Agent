"""Load DINOv2 and turn images into normalized patch features."""

import torch
import torch.nn.functional as functional
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from detector.config import IMAGE_SIZE, MODEL_NAME

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_processor = None
_model = None


def _load_model():
    """Load the image processor and model once, on the available device."""
    global _processor, _model
    if _model is None:
        _processor = AutoImageProcessor.from_pretrained(
            MODEL_NAME,
            size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
        )
        _model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    return _processor, _model


def extract_patch_features(images: list[Image.Image]) -> torch.Tensor:
    """Return L2-normalized patch tokens as [batch, patches, feature_dim]."""
    processor, model = _load_model()
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)

    with torch.inference_mode():
        # DINOv2 puts a whole-image CLS token before the individual patch tokens.
        patch_tokens = model(pixel_values=pixel_values).last_hidden_state[:, 1:, :]
        return functional.normalize(patch_tokens, p=2, dim=-1)
