"""Image and mask utilities for the ionogram-scaling skill."""

from __future__ import annotations

import base64
import io
import re
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


def load_and_resize(image_path: str, size: int) -> Image.Image:
    """Load an image. If size > 0, resize to size x size (square).
    If size == 0, keep original dimensions but cap at MAX_DIM on longest side."""
    img = Image.open(image_path).convert("RGB")
    if size and size > 0:
        return img.resize((size, size), Image.BILINEAR)
    # size == 0: keep aspect ratio, cap max dimension
    MAX_DIM = 1024
    w, h = img.size
    longest = max(w, h)
    if longest > MAX_DIM:
        scale = MAX_DIM / longest
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
    return img


def encode_base64(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
    """Encode a PIL image as base64. Defaults to JPEG for compact payloads."""
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        img.save(buf, format="JPEG", quality=quality)
    else:
        img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def data_url(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
    mime = f"image/{fmt.lower()}"
    return f"data:{mime};base64,{encode_base64(img, fmt, quality)}"


# ---------------------------------------------------------------------------
# Grid parsing
# ---------------------------------------------------------------------------

def parse_grid(text: str, grid_size: int) -> Optional[np.ndarray]:
    """Parse a binary grid from free-form VLM text into a (grid_size, grid_size)
    uint8 array. Returns None if parsing fails.
    """
    if text is None:
        return None

    # Try JSON first (array of arrays of ints, or array of strings).
    try:
        import json
        candidate = _extract_json_array(text)
        if candidate is not None:
            data = json.loads(candidate)
            grid = _coerce_grid(data, grid_size)
            if grid is not None:
                return grid
    except Exception:
        pass

    # Fallback: scan line by line, extract 0/1 tokens.
    rows: List[List[int]] = []
    for line in text.splitlines():
        line = line.strip().strip("[]")
        if not line:
            continue
        tokens = re.findall(r"[01]", line)
        if not tokens:
            continue
        rows.append([int(t) for t in tokens])
        if len(rows) >= grid_size:
            break

    if not rows:
        return None

    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    for r, row in enumerate(rows[:grid_size]):
        row = row[:grid_size]
        grid[r, : len(row)] = row
    return grid


def _extract_json_array(text: str) -> Optional[str]:
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _coerce_grid(data, grid_size: int) -> Optional[np.ndarray]:
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    if not isinstance(data, list):
        return None
    for r, row in enumerate(data[:grid_size]):
        if isinstance(row, str):
            tokens = re.findall(r"[01]", row)
            vals = [int(t) for t in tokens][:grid_size]
        elif isinstance(row, list):
            vals = [int(x) for x in row][:grid_size]
        else:
            return None
        grid[r, : len(vals)] = vals
    return grid


# ---------------------------------------------------------------------------
# Mask operations
# ---------------------------------------------------------------------------

def grid_to_mask(grid: np.ndarray, width: int, height: int) -> np.ndarray:
    """Upsample a binary grid to width x height using nearest neighbor."""
    g = grid.astype(np.uint8)
    img = Image.fromarray(g * 255, mode="L").resize((width, height), Image.NEAREST)
    return (np.array(img) > 127).astype(np.uint8)


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def mask_center_of_mass(mask: np.ndarray) -> Tuple[int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0, 0)
    return (int(round(xs.mean())), int(round(ys.mean())))


def overlay_masks(
    image: Image.Image,
    masks: List[np.ndarray],
    colors: Optional[List[Tuple[int, int, int]]] = None,
    alpha: float = 0.45,
) -> Image.Image:
    """Draw semi-transparent mask regions on top of `image`."""
    base = image.convert("RGB").copy()
    base_arr = np.array(base, dtype=np.uint8)

    if colors is None:
        palette = [
            (255, 80, 80), (80, 200, 80), (80, 120, 255),
            (255, 200, 0), (200, 80, 255), (0, 220, 220),
        ]
        colors = [palette[i % len(palette)] for i in range(len(masks))]

    blend = base_arr.astype(np.float32)
    for mask, color in zip(masks, colors):
        m = mask > 0
        for ch in range(3):
            blend[:, :, ch] = np.where(
                m,
                blend[:, :, ch] * (1 - alpha) + color[ch] * alpha,
                blend[:, :, ch],
            )

    return Image.fromarray(blend.astype(np.uint8), mode="RGB")
