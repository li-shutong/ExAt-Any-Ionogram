"""The ionogram-scaling skill agents (polyline tracing + code-based verification)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ionogram_scaling.config import Config
from ionogram_scaling.prompts import (
    build_critic_system,
    build_critic_user,
    build_describer_system,
    build_describer_user,
    build_tracer_system,
    build_tracer_user,
)
from ionogram_scaling.vlm_client import VLMClient, extract_json

logger = logging.getLogger("ionogram_scaling.agent")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Description:
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.raw, ensure_ascii=False, indent=2)


@dataclass
class TraceResult:
    """Polyline points output by the tracer."""
    polylines: List[Dict[str, Any]] = field(default_factory=list)
    total_points: int = 0
    confidence: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.raw, ensure_ascii=False, indent=2)

    def all_points(self) -> List[Tuple[float, float]]:
        pts = []
        for pl in self.polylines:
            for p in pl.get("points", []):
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    pts.append((float(p[0]), float(p[1])))
        return pts

    def points_str(self) -> str:
        """Compact string representation of all points for feedback."""
        lines = []
        for i, pl in enumerate(self.polylines):
            pts = pl.get("points", [])
            pts_str = ", ".join(f"[{p[0]:.0f},{p[1]:.0f}]" for p in pts)
            lines.append(f"  {pl.get('label', f'trace_{i}')}: [{pts_str}]")
        return "\n".join(lines)


@dataclass
class CriticResult:
    """VLM critic's judgment of the trace overlay."""
    confidence: float = 0.0
    issues: List[str] = field(default_factory=list)
    overall_assessment: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.raw, ensure_ascii=False, indent=2)

    def feedback_text(self) -> str:
        """Format feedback for the next tracer iteration."""
        parts = [f"置信度: {self.confidence:.2f}"]
        if self.overall_assessment:
            parts.append(f"总评: {self.overall_assessment}")
        if self.issues:
            parts.append("问题:")
            for iss in self.issues:
                parts.append(f"  - {iss}")
        return "\n".join(parts)


@dataclass
class PointVerification:
    """Per-point verification result from code-based pixel check."""
    point: Tuple[float, float]
    brightness: float       # 0-255, local pixel brightness
    is_echo: bool           # True if brightness above threshold
    confidence: float       # 0-1, normalized brightness score


@dataclass
class Verification:
    point_verifications: List[PointVerification] = field(default_factory=list)
    overall_confidence: float = 0.0
    n_passed: int = 0
    n_total: int = 0
    decision: str = "REJECT"


# ---------------------------------------------------------------------------
# Agent 1 — IonoDescriber (lightweight)
# ---------------------------------------------------------------------------

class IonoDescriber:
    def __init__(self, client: VLMClient, config: Config):
        self.client = client
        self.config = config

    def run(self, image_b64: str) -> Description:
        sys_prompt = build_describer_system(self.config)
        user_prompt = build_describer_user()
        raw = self.client.chat(sys_prompt, user_prompt, image_b64=image_b64)
        data = extract_json(raw)
        if data is None:
            logger.error("Describer returned unparseable output:\n%s", raw[:500])
            return Description()
        return Description(raw=data)


# ---------------------------------------------------------------------------
# Agent 2 — IonoTracer (VLM: outputs polyline points in pixel coords)
# ---------------------------------------------------------------------------

class IonoTracer:
    def __init__(self, client: VLMClient, config: Config):
        self.client = client
        self.config = config

    def run(self, image_b64: str, description_json: str,
            img_w: int, img_h: int,
            feedback: str = "", prev_points: str = "") -> Optional[TraceResult]:
        sys_prompt = build_tracer_system(self.config, img_w, img_h)
        user_prompt = build_tracer_user(description_json, img_w, img_h, feedback, prev_points)
        raw = self.client.chat(sys_prompt, user_prompt, image_b64=image_b64)
        data = extract_json(raw)
        if data is None:
            logger.error("Tracer returned unparseable output:\n%s", raw[:500])
            return None
        polylines = data.get("polylines", [])
        if not polylines:
            logger.warning("Tracer returned no polylines")
            return None
        total = sum(len(pl.get("points", [])) for pl in polylines)
        return TraceResult(
            polylines=polylines,
            total_points=total,
            confidence=float(data.get("confidence", 0.0)),
            raw=data,
        )


# ---------------------------------------------------------------------------
# Agent 2b — IonoCritic (VLM: judges trace overlay, describes what's wrong)
# ---------------------------------------------------------------------------

class IonoCritic:
    """VLM critic: looks at the overlay image (trace points on ionogram)
    and judges whether points are on the echo, describing specific issues."""

    def __init__(self, client: VLMClient, config: Config):
        self.client = client
        self.config = config

    def run(self, overlay_b64: str, description_json: str = "",
            n_polylines: int = 0,
            polyline_info: str = "") -> Optional[CriticResult]:
        sys_prompt = build_critic_system(self.config)
        user_prompt = build_critic_user(description_json, n_polylines, polyline_info)
        raw = self.client.chat(sys_prompt, user_prompt, image_b64=overlay_b64)
        data = extract_json(raw)
        if data is None:
            logger.error("Critic returned unparseable output:\n%s", raw[:500])
            return None
        return CriticResult(
            confidence=float(data.get("confidence", 0.0)),
            issues=data.get("issues", []),
            overall_assessment=data.get("overall_assessment", ""),
            raw=data if isinstance(data, dict) else {},
        )


# ---------------------------------------------------------------------------
# Agent 3 — PixelVerifier (CODE: checks brightness at each point)
# ---------------------------------------------------------------------------

class PixelVerifier:
    """Verifies tracer points by checking pixel brightness at each point location.
    Pure verification — does not move points. Uses VLM coordinates as-is.
    """

    def __init__(self, config: Config):
        self.config = config

    def run(self, image: Image.Image, trace: TraceResult) -> Verification:
        img_gray = np.array(image.convert("L"))  # (H, W) uint8 0-255
        h, w = img_gray.shape

        point_verifs: List[PointVerification] = []
        all_pts = trace.all_points()

        for (px, py) in all_pts:
            ix = int(np.clip(px, 0, w - 1))
            iy = int(np.clip(py, 0, h - 1))

            # Sample a 5x5 window, take max brightness (echo lines are thin)
            r = 2
            y0, y1 = max(0, iy - r), min(h, iy + r + 1)
            x0, x1 = max(0, ix - r), min(w, ix + r + 1)
            window = img_gray[y0:y1, x0:x1]
            brightness = float(window.max()) if window.size > 0 else 0.0

            is_echo = brightness >= self.config.echo_brightness_threshold
            conf = max(0.0, min(1.0, (brightness - 60.0) / 195.0))

            point_verifs.append(PointVerification(
                point=(px, py), brightness=brightness,
                is_echo=is_echo, confidence=conf,
            ))

        n_passed = sum(1 for pv in point_verifs if pv.is_echo)
        n_total = len(point_verifs)
        overall = (n_passed / n_total) if n_total > 0 else 0.0
        decision = "PASS" if overall >= self.config.confidence_threshold else "REVISION_NEEDED"

        return Verification(
            point_verifications=point_verifs,
            overall_confidence=overall,
            n_passed=n_passed,
            n_total=n_total,
            decision=decision,
        )


# ---------------------------------------------------------------------------
# Agent 4 — MaskBuilder (pure code: draws thick line through verified points)
# ---------------------------------------------------------------------------

class MaskBuilder:
    """Builds a binary mask by drawing thick lines through traced echo points."""

    def __init__(self, config: Config):
        self.config = config

    def run(self, image: Image.Image, trace: TraceResult) -> np.ndarray:
        w, h = image.size
        mask = np.zeros((h, w), dtype=np.uint8)

        from PIL import ImageDraw
        mask_img = Image.fromarray(mask * 255, mode="L")
        draw = ImageDraw.Draw(mask_img)

        # Use VLM's polyline points directly (pure LLM, no pixel correction)
        for pl in trace.polylines:
            pts = pl.get("points", [])
            if not pts:
                continue
            poly_pts = []
            for p in pts:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    poly_pts.append((float(p[0]), float(p[1])))

            if len(poly_pts) >= 2:
                line_width = self.config.mask_line_width
                draw.line(poly_pts, fill=255, width=line_width, joint="curve")
                for (px, py) in poly_pts:
                    r = line_width // 2
                    draw.ellipse([px - r, py - r, px + r, py + r], fill=255)

        mask = (np.array(mask_img) > 127).astype(np.uint8)
        return mask
