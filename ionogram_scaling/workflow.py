"""End-to-end orchestration of the ionogram-scaling skill workflow."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from ionogram_scaling.agents import (
    CriticResult,
    Description,
    IonoCritic,
    IonoDescriber,
    IonoTracer,
    MaskBuilder,
    TraceResult,
)
from ionogram_scaling.config import Config
from ionogram_scaling.image_utils import (
    encode_base64,
    load_and_resize,
    mask_bbox,
    mask_center_of_mass,
    overlay_masks,
)
from ionogram_scaling.vlm_client import VLMClient

logger = logging.getLogger("ionogram_scaling.workflow")


class IonoSegWorkflow:
    def __init__(self, config: Config):
        self.config = config
        client = VLMClient(config)
        self.describer = IonoDescriber(client, config)
        self.tracer = IonoTracer(client, config)
        self.critic = IonoCritic(client, config)
        self.mask_builder = MaskBuilder(config)

    # ------------------------------------------------------------------
    def run(self, image_path: str) -> Dict[str, Any]:
        cfg = self.config
        logger.info("Loading image: %s", image_path)
        img = load_and_resize(image_path, cfg.image_size)
        img_w, img_h = img.size
        img_b64 = encode_base64(img, fmt="JPEG", quality=cfg.jpeg_quality)
        logger.info("Image: %dx%d JPEG q%d -> %.1f KB",
                     img_w, img_h, cfg.jpeg_quality, len(img_b64) / 1024)

        out_dir = os.path.join(cfg.output_dir, os.path.splitext(os.path.basename(image_path))[0])
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(image_path))[0]

        # --- Agent 1: Describe ---
        logger.info("Agent 1 (Describer) running ...")
        description = self.describer.run(img_b64)
        self._save_json(description.to_json(), out_dir, stem, "_description.json")
        logger.info("Describer done: %s",
                     description.raw.get("shape_description", ""))

        # --- PDCA loop: Trace -> Critic -> Feedback -> Re-trace ---
        trace, critic_result = self._pdca_loop(
            img, img_b64, description, img_w, img_h, out_dir, stem
        )

        # --- Agent 4: Build mask from best trace ---
        min_confidence = 0.50  # Below this, the trace is too unreliable to output
        critic_conf = critic_result.confidence if critic_result else 0.0
        passed = critic_conf >= min_confidence

        logger.info("Agent 4 (MaskBuilder) running ...")
        if trace is not None and trace.total_points > 0 and passed:
            mask = self.mask_builder.run(img, trace)
            logger.info("Mask built (critic confidence %.2f >= %.2f)", critic_conf, min_confidence)
        else:
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            if not passed:
                logger.warning("Critic confidence %.2f < %.2f -> NOT outputting mask (failed)",
                               critic_conf, min_confidence)

        # Save mask + overlay
        mp = os.path.join(out_dir, f"{stem}_mask.png")
        Image.fromarray(mask * 255, mode="L").save(mp)
        logger.info("Saved mask: %s", mp)

        overlay = overlay_masks(img, [mask], colors=[(255, 80, 80)])
        op = os.path.join(out_dir, f"{stem}_overlay.png")
        overlay.save(op)
        logger.info("Saved overlay: %s", op)

        # Final JSON
        bbox = mask_bbox(mask) if mask.sum() > 0 else None
        com = mask_center_of_mass(mask) if mask.sum() > 0 else (0, 0)
        result = {
            "source_image": image_path,
            "image_size": [img_w, img_h],
            "confidence_threshold": cfg.confidence_threshold,
            "description": description.raw,
            "trace": trace.raw if trace else None,
            "critic": critic_result.raw if critic_result else None,
            "mask_info": {
                "bounding_box": list(bbox) if bbox else None,
                "center_of_mass": list(com),
                "pixel_count": int(mask.sum()),
            },
            "summary": {
                "passed": passed,
                "critic_confidence": critic_result.confidence if critic_result else 0.0,
                "n_points": trace.total_points if trace else 0,
                "n_polylines": len(trace.polylines) if trace else 0,
            },
        }
        jp = os.path.join(out_dir, f"{stem}_result.json")
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info("Saved result JSON: %s", jp)

        return result

    # ------------------------------------------------------------------
    def _pdca_loop(
        self, img: Image.Image, img_b64: str, description: Description,
        img_w: int, img_h: int, out_dir: str, stem: str
    ) -> Tuple[Optional[TraceResult], Optional[CriticResult]]:
        """Pure VLM PDCA: Plan(trace) -> Do(overlay) -> Check(critic) -> Act(feedback)."""
        cfg = self.config
        desc_json = description.to_json()

        best_trace = None
        best_critic = None
        best_conf = -1.0
        feedback = ""
        prev_points = ""

        for iteration in range(1, cfg.max_retries + 1):
            logger.info("=== PDCA iteration %d/%d ===", iteration, cfg.max_retries)

            # --- P: Trace ---
            logger.info("Agent 2 (Tracer), iteration %d ...", iteration)
            trace = self.tracer.run(
                img_b64, desc_json, img_w, img_h,
                feedback=feedback, prev_points=prev_points,
            )
            if trace is None:
                logger.warning("No trace produced on iteration %d", iteration)
                continue
            logger.info("Tracer: %d polylines, %d points",
                         len(trace.polylines), trace.total_points)
            self._save_json(trace.to_json(), out_dir, stem, f"_trace_iter{iteration}.json")

            # --- D: Build overlay for critic ---
            overlay_img = self._draw_points_overlay(img, trace)
            op = os.path.join(out_dir, f"{stem}_critic_input_iter{iteration}.png")
            overlay_img.save(op)
            overlay_b64 = encode_base64(overlay_img, fmt="JPEG", quality=cfg.jpeg_quality)

            # --- C: Critic judges ---
            logger.info("Agent 3 (Critic), iteration %d ...", iteration)
            polyline_info = self._format_polyline_info(trace)
            critic = self.critic.run(overlay_b64, desc_json,
                                     len(trace.polylines), polyline_info)
            if critic is None:
                critic = CriticResult(confidence=0.0, issues=["Critic parse failed"])
            self._save_json(critic.to_json(), out_dir, stem, f"_critic_iter{iteration}.json")
            logger.info("Critic: confidence=%.2f, assessment=%s",
                         critic.confidence, critic.overall_assessment)
            for iss in critic.issues:
                logger.info("  issue: %s", iss)

            # Track best by critic confidence
            if critic.confidence > best_conf:
                best_conf = critic.confidence
                best_trace = trace
                best_critic = critic

            # --- A: Check if good enough ---
            if critic.confidence >= cfg.confidence_threshold:
                logger.info("Critic confidence %.2f >= threshold %.2f -> ACCEPT",
                             critic.confidence, cfg.confidence_threshold)
                break

            # --- A: Prepare feedback for next iteration ---
            feedback = critic.feedback_text()
            prev_points = trace.points_str()
            logger.info("Feeding critic feedback to next iteration...")

        if best_conf < cfg.confidence_threshold:
            logger.warning("PDCA exhausted %d iterations, best confidence %.2f < threshold %.2f -> NOT ACCEPTED",
                           cfg.max_retries, best_conf, cfg.confidence_threshold)
        logger.info("Best critic confidence: %.2f", best_conf)
        return (best_trace, best_critic)

    # ------------------------------------------------------------------
    @staticmethod
    def _format_polyline_info(trace: TraceResult) -> str:
        """Format polyline labels and point counts for the critic."""
        lines = []
        for i, pl in enumerate(trace.polylines):
            label = pl.get("label", f"trace_{i+1}")
            n = len(pl.get("points", []))
            lines.append(f"  第{i+1}条: label={label}, {n}个点")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Distinct colors for each polyline so the critic can visually count them
    POLYLINE_COLORS = [
        (255, 0, 0),      # red
        (0, 255, 0),      # green
        (0, 100, 255),    # blue
        (255, 255, 0),    # yellow
        (255, 0, 255),    # magenta
        (0, 255, 255),    # cyan
        (255, 165, 0),    # orange
        (128, 0, 128),    # purple
    ]

    def _draw_points_overlay(self, img: Image.Image, trace: TraceResult) -> Image.Image:
        """Draw each polyline in a DIFFERENT color so the critic can count them."""
        overlay = img.convert("RGBA")
        draw = ImageDraw.Draw(overlay)
        for i, pl in enumerate(trace.polylines):
            color = self.POLYLINE_COLORS[i % len(self.POLYLINE_COLORS)]
            pts = pl.get("points", [])
            poly_pts = [(float(p[0]), float(p[1])) for p in pts
                        if isinstance(p, (list, tuple)) and len(p) >= 2]
            if len(poly_pts) >= 2:
                draw.line(poly_pts, fill=color + (160,), width=2, joint="curve")
            for (px, py) in poly_pts:
                r = 3
                draw.ellipse([px - r, py - r, px + r, py + r],
                             fill=color + (200,), outline=(255, 255, 255, 255))
        return overlay.convert("RGB")

    # ------------------------------------------------------------------
    @staticmethod
    def _save_json(text: str, out_dir: str, stem: str, suffix: str) -> None:
        p = os.path.join(out_dir, f"{stem}{suffix}")
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info("Saved: %s", p)
