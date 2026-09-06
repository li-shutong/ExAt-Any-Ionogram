#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdsa.py - 分析三站点 PDSA (PDCA) 迭代效果

查找 {daofu_50, zhangye_50, puer_50} 中存在 trace_iter1/2/3 三张结果图的样本,
把每一轮预测与对应 GT 做匹配, 输出每轮精度统计。

输出:
  - <paper>/pdsa_eval.json
  - <paper>/pdsa_eval.csv
  - 控制台汇总
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "..", "Paper")

N_GRID = 200
Y_AXIS_KM = 560.0
X_AXIS_MHZ = 20.0
ITERATIONS = [1, 2, 3]


def _stem_strip_jpg(s: str) -> str:
    while s.endswith(".jpg"):
        s = s[: -len(".jpg")]
    return s


def load_gt(gt_dir):
    """{stem: (img_path, json_path, polylines)}"""
    out = {}
    if not os.path.isdir(gt_dir):
        return out
    for name in os.listdir(gt_dir):
        if not name.endswith(".json"):
            continue
        stem = _stem_strip_jpg(name[: -len(".json")])
        jp = os.path.join(gt_dir, name)
        ip = os.path.join(gt_dir, stem + ".jpg.jpg")
        if not os.path.exists(ip):
            continue
        try:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
            out[stem] = (ip, jp, data.get("polylines", []))
        except (OSError, json.JSONDecodeError) as e:
            print(f"[skip GT] {jp}: {e}")
    return out


def img_size(path):
    try:
        with Image.open(path) as im:
            return im.size
    except Exception as e:
        print(f"[img size] {path}: {e}")
        return None


def rightmost_polyline(polylines):
    cand = []
    for p in polylines:
        pts = p.get("points", [])
        if not pts:
            continue
        xs = [pt[0] for pt in pts]
        cand.append((max(xs), sum(xs) / len(xs), p))
    if not cand:
        return None
    cand.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return cand[0][2]


def to_array(poly):
    if poly is None:
        return None
    pts = poly.get("points", [])
    if len(pts) < 2:
        return None
    return np.asarray(pts, dtype=float)


def normalize(arr, orig_w, orig_h):
    out = arr.copy()
    out[:, 0] /= orig_w
    out[:, 1] /= orig_h
    return out


def pred_to_orig(arr, pred_w, pred_h, orig_w, orig_h):
    out = arr.copy()
    out[:, 0] *= orig_w / pred_w
    out[:, 1] *= orig_h / pred_h
    return out


def interp_curve(arr_norm):
    if arr_norm is None or len(arr_norm) < 2:
        return None
    x = arr_norm[:, 0]
    y = arr_norm[:, 1]
    order = np.argsort(x)
    x, y = x[order], y[order]
    xs, idx = np.unique(x, return_inverse=True)
    if len(xs) < 2:
        return None
    ys = np.zeros_like(xs)
    cnt = np.zeros_like(xs)
    for i, ix in enumerate(idx):
        ys[ix] += y[i]
        cnt[ix] += 1
    ys /= cnt
    return xs, ys


def evaluate(gt_arr, pred_arr, orig_w, orig_h, pred_w, pred_h):
    gt_norm = normalize(gt_arr, orig_w, orig_h)
    pred_norm = normalize(pred_to_orig(pred_arr, pred_w, pred_h, orig_w, orig_h), orig_w, orig_h)

    gi = interp_curve(gt_norm)
    pi = interp_curve(pred_norm)
    if gi is None or pi is None:
        return None
    gx, gy = gi
    px, py = pi

    x_lo = max(float(gx[0]), float(px[0]))
    x_hi = min(float(gx[-1]), float(px[-1]))
    if x_hi - x_lo < 1e-6:
        return None

    x_common = np.linspace(x_lo, x_hi, N_GRID)
    y_gt = np.interp(x_common, gx, gy)
    y_pred = np.interp(x_common, px, py)
    y_err = y_gt - y_pred

    rmse = float(np.sqrt(np.mean(y_err ** 2))) * Y_AXIS_KM
    mae = float(np.mean(np.abs(y_err))) * Y_AXIS_KM
    max_abs = float(np.max(np.abs(y_err))) * Y_AXIS_KM
    mean_err = float(np.mean(y_err)) * Y_AXIS_KM
    bias_km = mean_err
    std_km = float(np.std(y_err)) * Y_AXIS_KM

    if np.std(y_gt) > 1e-12 and np.std(y_pred) > 1e-12:
        pearson_r = float(np.corrcoef(y_gt, y_pred)[0, 1])
        pearson_r2 = pearson_r * pearson_r
    else:
        pearson_r = float("nan")
        pearson_r2 = float("nan")

    foF2_gt_mhz = float(gx[-1]) * X_AXIS_MHZ
    foF2_pred_mhz = float(px[-1]) * X_AXIS_MHZ
    foF2_err_mhz = foF2_pred_mhz - foF2_gt_mhz

    frac = 0.1
    gt_thr = gx[0] + frac * (gx[-1] - gx[0])
    pred_thr = px[0] + frac * (px[-1] - px[0])
    hF2_gt_km = float(np.mean(gy[gx <= gt_thr])) * Y_AXIS_KM
    hF2_pred_km = float(np.mean(py[px <= pred_thr])) * Y_AXIS_KM
    hF2_err_km = hF2_pred_km - hF2_gt_km

    return {
        "x_grid": x_common.tolist(),
        "y_gt": y_gt.tolist(),
        "y_pred": y_pred.tolist(),
        "y_err": y_err.tolist(),
        "rmse_km": rmse,
        "mae_km": mae,
        "max_abs_err_km": max_abs,
        "mean_err_km": mean_err,
        "bias_km": bias_km,
        "std_km": std_km,
        "pearson_r": pearson_r,
        "pearson_r2": pearson_r2,
        "foF2_gt_mhz": foF2_gt_mhz,
        "foF2_pred_mhz": foF2_pred_mhz,
        "foF2_err_mhz": foF2_err_mhz,
        "hF2_gt_km": hF2_gt_km,
        "hF2_pred_km": hF2_pred_km,
        "hF2_err_km": hF2_err_km,
        "x_range_common": [x_lo, x_hi],
        "x_coverage": float(x_hi - x_lo),
        "n_eval_points": N_GRID,
    }


def load_iter_trace(pred_dir, stem, iteration):
    """加载某一轮的 trace_iter{iteration}.json"""
    sub = stem + ".jpg"
    rj = os.path.join(pred_dir, sub, f"{sub}_trace_iter{iteration}.json")
    if not os.path.exists(rj):
        return None
    try:
        with open(rj, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("polylines", [])
    except (OSError, json.JSONDecodeError) as e:
        print(f"[skip iter] {rj}: {e}")
        return None


def find_three_round_images(pred_dir):
    """返回存在 iter1/2/3 的 stem 列表"""
    out = []
    if not os.path.isdir(pred_dir):
        return out
    for name in os.listdir(pred_dir):
        sub = os.path.join(pred_dir, name)
        if not os.path.isdir(sub):
            continue
        stem = _stem_strip_jpg(name)
        if all(
            os.path.exists(os.path.join(sub, f"{name}_trace_iter{i}.json"))
            for i in ITERATIONS
        ):
            out.append(stem)
    return sorted(out)


def eval_station(station, gt_dir, pred_dir):
    gt_map = load_gt(gt_dir)
    stems = find_three_round_images(pred_dir)
    common = [s for s in stems if s in gt_map]

    records = []
    skipped = []
    for stem in common:
        gt_img, gt_json, gt_polys = gt_map[stem]
        gt_poly = rightmost_polyline(gt_polys)
        if gt_poly is None:
            skipped.append((stem, "no GT polyline"))
            continue
        gt_arr = to_array(gt_poly)
        if gt_arr is None:
            skipped.append((stem, "GT too few points"))
            continue

        wh = img_size(gt_img)
        if wh is None:
            skipped.append((stem, "gt img size fail"))
            continue
        gw, gh = wh

        pred_img = os.path.join(pred_dir, stem + ".jpg.jpg")
        result_json = os.path.join(pred_dir, stem + ".jpg", stem + ".jpg_result.json")
        pw, ph = gw, gh
        if os.path.exists(result_json):
            try:
                with open(result_json, "r", encoding="utf-8") as f:
                    res = json.load(f)
                size = res.get("image_size", [])
                if len(size) >= 2:
                    pw, ph = size[0], size[1]
            except (OSError, json.JSONDecodeError):
                pass
        elif os.path.exists(pred_img):
            wh2 = img_size(pred_img)
            if wh2 is not None:
                pw, ph = wh2

        for iteration in ITERATIONS:
            pred_polys = load_iter_trace(pred_dir, stem, iteration)
            pred_poly = rightmost_polyline(pred_polys) if pred_polys else None
            if pred_poly is None:
                skipped.append((stem, f"iter{iteration} no polyline"))
                continue
            pred_arr = to_array(pred_poly)
            if pred_arr is None:
                skipped.append((stem, f"iter{iteration} too few points"))
                continue

            m = evaluate(gt_arr, pred_arr, gw, gh, pw, ph)
            if m is None:
                skipped.append((stem, f"iter{iteration} interp fail"))
                continue

            rec = {
                "stem": stem,
                "station": station,
                "iteration": iteration,
                "gt_image": gt_img,
                "gt_json": gt_json,
                "gt_label": gt_poly.get("label"),
                "gt_n_points": int(len(gt_arr)),
                "pred_image": pred_img,
                "pred_label": pred_poly.get("label"),
                "pred_n_points": int(len(pred_arr)),
                "orig_image_size": [gw, gh],
                "pred_image_size": [pw, ph],
                **m,
            }
            records.append(rec)

    info = {
        "n_gt": len(gt_map),
        "n_three_round": len(stems),
        "n_common": len(common),
        "n_evaluated_records": len(records),
        "skipped": skipped,
    }
    return records, info


def compute_offsets_per_group(records):
    """按 (station, iteration) 分别计算 foF2 / h'F2 系统偏移并生成 corrected 字段"""
    groups = defaultdict(list)
    for r in records:
        groups[(r["station"], r["iteration"])].append(r)

    offsets = {}
    for key, recs in groups.items():
        if not recs:
            offsets[key] = {"fof2_mhz": 0.0, "hf2_km": 0.0}
            continue
        fof2_off = sum(r["foF2_err_mhz"] for r in recs) / len(recs)
        hf2_off = sum(r["hF2_err_km"] for r in recs) / len(recs)
        for r in recs:
            r["foF2_err_corrected_mhz"] = r["foF2_err_mhz"] - fof2_off
            r["hF2_err_corrected_km"] = r["hF2_err_km"] - hf2_off
        offsets[key] = {"fof2_mhz": fof2_off, "hf2_km": hf2_off}
    return offsets


def aggregate(records):
    """按 (station, iteration) 聚合均值"""
    groups = defaultdict(list)
    for r in records:
        groups[(r["station"], r["iteration"])].append(r)

    out = {}
    for key, recs in groups.items():
        rmse = np.array([r["rmse_km"] for r in recs])
        r2 = np.array([r["pearson_r2"] for r in recs if r["pearson_r2"] == r["pearson_r2"]])
        mean_err = np.array([r["mean_err_km"] for r in recs])

        foF2_err = np.array([r["foF2_err_mhz"] for r in recs])
        foF2_corr = np.array([r["foF2_err_corrected_mhz"] for r in recs])
        foF2_gt = np.array([r["foF2_gt_mhz"] for r in recs])
        foF2_pred = np.array([r["foF2_pred_mhz"] for r in recs])

        hF2_err = np.array([r["hF2_err_km"] for r in recs])
        hF2_corr = np.array([r["hF2_err_corrected_km"] for r in recs])
        hF2_gt = np.array([r["hF2_gt_km"] for r in recs])
        hF2_pred = np.array([r["hF2_pred_km"] for r in recs])

        foF2_rmse = float(np.sqrt(np.mean(foF2_err ** 2)))
        foF2_rmse_corr = float(np.sqrt(np.mean(foF2_corr ** 2)))
        hF2_rmse = float(np.sqrt(np.mean(hF2_err ** 2)))
        hF2_rmse_corr = float(np.sqrt(np.mean(hF2_corr ** 2)))

        if np.std(foF2_gt) > 1e-12 and np.std(foF2_pred) > 1e-12:
            foF2_r2 = float(np.corrcoef(foF2_gt, foF2_pred)[0, 1] ** 2)
        else:
            foF2_r2 = float("nan")
        if np.std(hF2_gt) > 1e-12 and np.std(hF2_pred) > 1e-12:
            hF2_r2 = float(np.corrcoef(hF2_gt, hF2_pred)[0, 1] ** 2)
        else:
            hF2_r2 = float("nan")

        out[key] = {
            "n": len(recs),
            "rmse_km_mean": float(np.mean(rmse)),
            "mae_km_mean": float(np.mean([r["mae_km"] for r in recs])),
            "max_abs_err_km_mean": float(np.mean([r["max_abs_err_km"] for r in recs])),
            "mean_err_km_mean": float(np.mean(mean_err)),
            "bias_km_mean": float(np.mean([r["bias_km"] for r in recs])),
            "std_km_mean": float(np.mean([r["std_km"] for r in recs])),
            "pearson_r_mean": float(np.nanmean([r["pearson_r"] for r in recs])),
            "pearson_r2_mean": float(np.nanmean(r2)) if len(r2) else float("nan"),
            "foF2_err_mhz_mean": float(np.mean(foF2_err)),
            "foF2_rmse_mhz": foF2_rmse,
            "foF2_rmse_corrected_mhz": foF2_rmse_corr,
            "foF2_r2": foF2_r2,
            "hF2_err_km_mean": float(np.mean(hF2_err)),
            "hF2_rmse_km": hF2_rmse,
            "hF2_rmse_corrected_km": hF2_rmse_corr,
            "hF2_r2": hF2_r2,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="分析三站点 PDCA 迭代效果")
    parser.add_argument("--paper", type=str, default=PAPER,
                        help="Paper 数据根目录 (默认: skill/../Paper)")
    parser.add_argument("--out-json", type=str, default=None,
                        help="输出 JSON 路径 (默认: <paper>/pdsa_eval.json)")
    parser.add_argument("--out-csv", type=str, default=None,
                        help="输出 CSV 路径 (默认: <paper>/pdsa_eval.csv)")
    args = parser.parse_args()

    paper = os.path.abspath(args.paper)
    out_json = args.out_json or os.path.join(paper, "pdsa_eval.json")
    out_csv = args.out_csv or os.path.join(paper, "pdsa_eval.csv")

    stations = [
        ("daofu",   os.path.join(paper, "daofu_labeled"),   os.path.join(paper, "daofu_50")),
        ("zhangye", os.path.join(paper, "zhangye_labeled"), os.path.join(paper, "zhangye_50")),
        ("puer",    os.path.join(paper, "puer_labeled"),    os.path.join(paper, "puer_50")),
    ]

    out = {}
    station_infos = {}
    all_records = []

    for station, gt_dir, pred_dir in stations:
        recs, info = eval_station(station, gt_dir, pred_dir)
        out[station] = recs
        station_infos[station] = info
        all_records.extend(recs)
        print(f"[{station}] GT={info['n_gt']}  3-round={info['n_three_round']}  "
              f"common={info['n_common']}  records={info['n_evaluated_records']}  "
              f"skip={len(info['skipped'])}")

    offsets = compute_offsets_per_group(all_records)
    agg = aggregate(all_records)
    agg_serializable = {f"{station}_iter{iteration}": v
                        for (station, iteration), v in agg.items()}
    offsets_serializable = {f"{station}_iter{iteration}": v
                            for (station, iteration), v in offsets.items()}

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {"stations": out, "info": station_infos,
             "offsets_per_station_iteration": offsets_serializable,
             "aggregate": agg_serializable},
            f, ensure_ascii=False, indent=2
        )
    print(f"\nJSON: {out_json}")

    cols = [
        "station", "stem", "iteration",
        "gt_label", "pred_label",
        "rmse_km", "mae_km", "max_abs_err_km", "mean_err_km",
        "bias_km", "std_km",
        "pearson_r", "pearson_r2",
        "foF2_gt_mhz", "foF2_pred_mhz", "foF2_err_mhz", "foF2_err_corrected_mhz",
        "hF2_gt_km", "hF2_pred_km", "hF2_err_km", "hF2_err_corrected_km",
        "x_coverage", "gt_n_points", "pred_n_points",
        "gt_image", "pred_image",
    ]
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in all_records:
            w.writerow([r.get(c, "") for c in cols])
    print(f"CSV : {out_csv}")

    print("\n=== 按站点/轮次汇总 ===")
    for station, _, _ in stations:
        print(f"\n[{station}]")
        for iteration in ITERATIONS:
            stats = agg.get((station, iteration))
            if stats is None:
                print(f"  iter{iteration}: 无数据")
                continue
            print(
                f"  iter{iteration}  n={stats['n']}  "
                f"RMSE={stats['rmse_km_mean']:.2f}km  "
                f"MAE={stats['mae_km_mean']:.2f}km  "
                f"r={stats['pearson_r_mean']:.3f}  r2={stats['pearson_r2_mean']:.4f}"
            )
            print(
                f"           "
                f"foF2_rmse={stats['foF2_rmse_mhz']:.3f}MHz"
                f"({stats['foF2_rmse_corrected_mhz']:.3f}corr)  "
                f"foF2_r2={stats['foF2_r2']:.4f}"
            )
            print(
                f"           "
                f"h'F2_rmse={stats['hF2_rmse_km']:.2f}km"
                f"({stats['hF2_rmse_corrected_km']:.2f}corr)  "
                f"h'F2_r2={stats['hF2_r2']:.4f}"
            )


if __name__ == "__main__":
    main()
