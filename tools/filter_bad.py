#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""筛选 foF2 误差 > 2MHz 或 h'F2 误差 > 100km 的离群记录, 输出重跑命令."""

from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "..", "Paper")


def main():
    parser = argparse.ArgumentParser(description="筛选离群评估记录")
    parser.add_argument("--paper", type=str, default=PAPER,
                        help="Paper 数据根目录 (默认: skill/../Paper)")
    parser.add_argument("--eval-json", type=str, default=None,
                        help="输入 test_eval.json 路径 (默认: <paper>/test_eval.json)")
    parser.add_argument("--fof2-thr", type=float, default=2.0,
                        help="foF2 误差阈值 (MHz)")
    parser.add_argument("--hf2-thr", type=float, default=100.0,
                        help="h'F2 误差阈值 (km)")
    args = parser.parse_args()

    paper = os.path.abspath(args.paper)
    eval_json = args.eval_json or os.path.join(paper, "test_eval.json")

    pred_dirs = {
        "daofu": os.path.join(paper, "daofu_50"),
        "zhangye": os.path.join(paper, "zhangye_50"),
        "puer": os.path.join(paper, "puer_50"),
    }

    with open(eval_json, encoding="utf-8") as f:
        data = json.load(f)

    bad = []
    for station, recs in data["stations"].items():
        for r in recs:
            fof2_err = abs(r.get("foF2_err_mhz", 0.0))
            hf2_err = abs(r.get("hF2_err_km", 0.0))
            if fof2_err > args.fof2_thr or hf2_err > args.hf2_thr:
                bad.append((
                    station, r["stem"],
                    r.get("rmse_km"), r.get("pearson_r2"),
                    r.get("foF2_err_mhz"), r.get("hF2_err_km"),
                ))

    bad.sort(key=lambda x: -max(abs(x[4] / args.fof2_thr), abs(x[5] / args.hf2_thr)))

    print(f"共 {len(bad)} 条 (|foF2_err|>{args.fof2_thr}MHz 或 |h2_err|>{args.hf2_thr}km):\n")
    for station, stem, rmse, r2, fof2, hf2 in bad:
        pred_dir = pred_dirs.get(station, "")
        img = os.path.join(pred_dir, stem + ".jpg.jpg")
        if not os.path.exists(img):
            img = os.path.join(pred_dir, stem + ".jpg")
        print(f"# {station}  {stem}")
        print(f"#   rmse={rmse:.1f}km  r2={r2:.3f}  "
              f"foF2_err={fof2:+.2f}MHz  hF2_err={hf2:+.1f}km")
        print(f'python -m ionogram_scaling.main --image "{img}" --output-dir "{pred_dir}"')
        print()


if __name__ == "__main__":
    main()
