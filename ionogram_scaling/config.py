"""Configuration for the ionogram-scaling skill.

All settings can be overridden via environment variables or CLI flags.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


@dataclass
class Config:
    # ---- VLM backend (OpenAI-compatible) ----
    api_key: str = field(
        default_factory=lambda: _env("VLM_API_KEY", "")
        or _env("OPENAI_API_KEY", "")
    )
    base_url: str = field(default_factory=lambda: _env("VLM_BASE_URL", ""))
    model: str = field(default_factory=lambda: _env("VLM_MODEL", ""))
    # "responses" -> OpenAI Responses API (/responses)
    # "chat"      -> Chat Completions API (/chat/completions)
    wire_api: str = field(default_factory=lambda: _env("VLM_WIRE_API", "chat"))
    http_headers: str = field(default_factory=lambda: _env("VLM_HTTP_HEADERS", ""))
    # Reasoning effort for reasoning models: "low" | "medium" | "high".
    reasoning_effort: str = field(default_factory=lambda: _env("VLM_REASONING_EFFORT", "low"))

    # ---- Image ----
    # 0 = use original image dimensions (no resize). >0 = resize to square.
    image_size: int = field(default_factory=lambda: _env_int("IMAGE_SIZE", 0))
    # JPEG quality for API calls (1-95).  Lower = smaller payload = fewer 502s.
    jpeg_quality: int = field(default_factory=lambda: _env_int("JPEG_QUALITY", 85))

    # ---- Mask grid ----
    grid_size: int = field(default_factory=lambda: _env_int("GRID_SIZE", 128))

    # ---- Mask ----
    # Line width (pixels) for drawing the mask through traced points.
    mask_line_width: int = field(default_factory=lambda: _env_int("MASK_LINE_WIDTH", 5))
    # Brightness threshold (0-255) for pixel-based point verification.
    echo_brightness_threshold: float = field(
        default_factory=lambda: _env_float("ECHO_BRIGHTNESS_THRESHOLD", 100.0)
    )
    # Snapping radius (pixels): each traced point is moved to the brightest
    # pixel within this radius, correcting VLM spatial offset.
    snap_radius: int = field(default_factory=lambda: _env_int("SNAP_RADIUS", 15))

    # ---- Workflow ----
    confidence_threshold: float = field(default_factory=lambda: _env_float("CONFIDENCE_THRESHOLD", 0.9))
    max_retries: int = field(default_factory=lambda: _env_int("MAX_RETRIES", 3))
    # API-level retry count for transient 5xx errors.
    api_retries: int = field(default_factory=lambda: _env_int("API_RETRIES", 6))
    max_tokens: int = field(default_factory=lambda: _env_int("VLM_MAX_TOKENS", 8192))

    # ---- Output ----
    output_dir: str = field(default_factory=lambda: _env("OUTPUT_DIR", "output"))

    def validate(self) -> Optional[str]:
        if not self.api_key:
            return "VLM_API_KEY is not set"
        if not self.base_url:
            return "VLM_BASE_URL is not set"
        if not self.model:
            return "VLM_MODEL is not set"
        if self.image_size < 0:
            return "image_size must be >= 0 (0 = original size)"
        if self.grid_size <= 0:
            return "grid_size must be positive"
        if not (0.0 < self.confidence_threshold <= 1.0):
            return "confidence_threshold must be in (0, 1]"
        if self.reasoning_effort not in ("low", "medium", "high"):
            return "reasoning_effort must be low, medium, or high"
        return None
