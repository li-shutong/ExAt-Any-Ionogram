"""Command-line entry point for the ionogram-scaling skill."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time

from ionogram_scaling.config import Config
from ionogram_scaling.workflow import IonoSegWorkflow

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ionogram-scaling: VLM multi-agent ionogram segmentation")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", type=str, help="Path to a single input ionogram image.")
    g.add_argument("--batch", type=str, help="Folder containing ionogram images to batch process.")
    p.add_argument("--image-size", type=int, default=None, help="Resize target (0=original).")
    p.add_argument("--threshold", type=float, default=None, help="Critic confidence threshold.")
    p.add_argument("--max-retries", type=int, default=None, help="Max PDCA iterations (default 3).")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory (default 'output').")
    p.add_argument("--model", type=str, default=None, help="VLM model name.")
    p.add_argument("--base-url", type=str, default=None, help="VLM API base URL.")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    if args.image_size is not None:
        cfg.image_size = args.image_size
    if args.threshold is not None:
        cfg.confidence_threshold = args.threshold
    if args.max_retries is not None:
        cfg.max_retries = args.max_retries
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.model is not None:
        cfg.model = args.model
    if args.base_url is not None:
        cfg.base_url = args.base_url
    return cfg


def collect_images(folder: str) -> list:
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(glob.glob(os.path.join(folder, ext)))
        images.extend(glob.glob(os.path.join(folder, ext.upper())))
    # Deduplicate and sort
    seen = set()
    unique = []
    for p in images:
        rp = os.path.realpath(p)
        if rp not in seen and os.path.isfile(p):
            seen.add(rp)
            unique.append(p)
    return sorted(unique)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = build_config(args)
    err = cfg.validate()
    if err:
        logging.error(err)
        return 1

    workflow = IonoSegWorkflow(cfg)

    # --- Single image mode ---
    if args.image:
        result = workflow.run(args.image)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    # --- Batch mode ---
    images = collect_images(args.batch)
    if not images:
        logging.error("No images found in %s", args.batch)
        return 1

    logging.info("Found %d images in %s", len(images), args.batch)
    results = []
    ok, fail = 0, 0

    for i, img_path in enumerate(images, 1):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        logging.info("=" * 60)
        logging.info("Batch %d/%d: %s", i, len(images), stem)
        logging.info("=" * 60)
        t0 = time.time()
        try:
            result = workflow.run(img_path)
            conf = result.get("summary", {}).get("critic_confidence", 0.0)
            npts = result.get("summary", {}).get("n_points", 0)
            logging.info("Done: %s | confidence=%.2f | points=%d | %.1fs",
                         stem, conf, npts, time.time() - t0)
            results.append({"image": stem, "status": "ok", "confidence": conf, "points": npts})
            ok += 1
        except Exception as e:
            logging.error("Failed: %s -> %s", stem, e)
            results.append({"image": stem, "status": "fail", "error": str(e)})
            fail += 1

    # Save batch summary
    summary_path = os.path.join(cfg.output_dir, "batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"total": len(images), "ok": ok, "fail": fail, "results": results},
                  f, ensure_ascii=False, indent=2)
    logging.info("Batch complete: %d ok, %d fail. Summary: %s", ok, fail, summary_path)
    print(json.dumps({"total": len(images), "ok": ok, "fail": fail}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
