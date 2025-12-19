from __future__ import annotations

import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional

from PIL import Image


TinkerImageFormat = Literal["png", "jpeg"]


@dataclass(frozen=True)
class PreparedImage:
    data: bytes
    format: TinkerImageFormat
    width: int
    height: int
    tokens: int


def estimate_image_tokens(*, width: int, height: int, patch_size: int = 28) -> int:
    """
    Roughly estimate vision tokens from image dimensions.

    Tinker expects a token count for each image chunk; this value is used for
    context-length accounting. We estimate tokens as the number of square patches.
    """
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    patches_w = max(1, math.ceil(width / patch_size))
    patches_h = max(1, math.ceil(height / patch_size))
    return patches_w * patches_h


def _infer_tinker_format(path: Path) -> Optional[TinkerImageFormat]:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "jpeg"
    if ext == ".png":
        return "png"
    return None


def prepare_image_for_tinker(
    image_path: Path,
    *,
    resize_long_edge: Optional[int] = None,
    jpeg_quality: int = 95,
    patch_size: int = 28,
) -> PreparedImage:
    """
    Load an image and return a Tinker-ready payload (bytes + metadata).

    If `resize_long_edge` is set and the image exceeds it, we resize with LANCZOS and
    re-encode as JPEG. If no resize is needed and the format is already supported,
    we keep the original bytes.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    original_format = _infer_tinker_format(image_path)
    with Image.open(image_path) as img:
        img.load()

        width, height = img.size
        needs_resize = resize_long_edge is not None and max(width, height) > resize_long_edge

        if not needs_resize and original_format is not None:
            data = image_path.read_bytes()
            tokens = estimate_image_tokens(width=width, height=height, patch_size=patch_size)
            return PreparedImage(
                data=data,
                format=original_format,
                width=width,
                height=height,
                tokens=tokens,
            )

        return prepare_pil_image_for_tinker(
            img,
            resize_long_edge=resize_long_edge,
            jpeg_quality=jpeg_quality,
            patch_size=patch_size,
        )


def prepare_pil_image_for_tinker(
    img: Image.Image,
    *,
    resize_long_edge: Optional[int] = None,
    jpeg_quality: int = 95,
    patch_size: int = 28,
) -> PreparedImage:
    """
    Encode a PIL image into a Tinker-ready JPEG payload.

    Used for cropped column images where we don't have a source file path.
    """
    width, height = img.size
    if img.mode != "RGB":
        img = img.convert("RGB")

    if resize_long_edge is not None and max(width, height) > resize_long_edge:
        ratio = resize_long_edge / max(width, height)
        new_w = max(1, int(round(width * ratio)))
        new_h = max(1, int(round(height * ratio)))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        width, height = img.size

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    out = buf.getvalue()
    tokens = estimate_image_tokens(width=width, height=height, patch_size=patch_size)
    return PreparedImage(data=out, format="jpeg", width=width, height=height, tokens=tokens)

