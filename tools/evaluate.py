#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py - 评估 VLM 描迹输出与 GT 的位置精度

对每对匹配图像 (GT vs 预测):
  1. 各自选"最右侧"那条描迹 (max x 最大者, 平手用 mean x 决胜)
  2. 把预测坐标 (在缩放图上) 还原到原图坐标系, 再归一化到 [0,1]
  3. 沿 x 轴线性插值到共同 x 网格, 得到 y_gt / y_pred 两个等长数列
  4. 误差 y_err = y_gt - y_pred (保留正负), 计算 R^2 / RMSE / MAE / 最大误差 / 平均误差
  5. 输出 JSON (按 daofu / zhangye / puer 三站点分组) + CSV 表格 + 控制台汇总

误差单位: km (纵轴归一化误差 × 560km, 560km = 整张图纵轴对应的物理高度)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image

# 默认相对于 skill/ 目录定位 Paper/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "..", "Paper")

N_GRID = 200
Y_AXIS_KM = 560.0
X_AXIS_MHZ = 20.0


def _stem_strip_jpg(s: str) -> str:
    while s.endswith(".jpg"):
        s = s[:-len(".jpg")]
    return s


def load_gt(gt_dir):
    """{stem: (img_path, json_path, polylines)}"""
    out = {}
    if not os.path.isdir(gt_dir):
        return out
    for name in os.listdir(gt_dir):
        if not name.endswith(".json"):
            continue
        stem = _stem_strip_jpg(name[:-len(".json")])
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


def load_pred(pred_dir):
    """{stem: (img_path, json_path, polylines, image_size)}"""
    out = {}
    if not os.path.isdir(pred_dir):
        return out
    for name in os.listdir(pred_dir):
        sub = os.path.join(pred_dir, name)
        if not os.path.isdir(sub):
            continue
        stem = _stem_strip_jpg(name)
        rj = os.path.join(sub, name + "_result.json")
        if not os.path.exists(rj):
            continue
        ip = os.path.join(pred_dir, name + ".jpg")
        try:
            with open(rj, "r", encoding="utf-8") as f:
                data = json.load(f)
            trace = data.get("trace", {}) or {}
            out[stem] = (ip, rj, trace.get("polylines", []),
                         data.get("image_size", []))
        except (OSError, json.JSONDecodeError) as e:
            print(f"[skip pred] {rj}: {e}")
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
    pred_norm = normalize(pred_to_orig(pred_arr, pred_w, pred_h,
                                       orig_w, orig_h), orig_w, orig_h)

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


def eval_station(station, gt_dir, pred_dir):
    gt_map = load_gt(gt_dir)
    pred_map = load_pred(pred_dir)
    common = sorted(set(gt_map) & set(pred_map))

    records = []
    skipped = []
    for stem in common:
        gt_img, gt_json, gt_polys = gt_map[stem]
        pred_img, pred_json, pred_polys, pred_size = pred_map[stem]

        gt_poly = rightmost_polyline(gt_polys)
        pred_poly = rightmost_polyline(pred_polys)
        if gt_poly is None or pred_poly is None:
            skipped.append((stem, "no polyline"))
            continue

        gt_arr = to_array(gt_poly)
        pred_arr = to_array(pred_poly)
        if gt_arr is None or pred_arr is None:
            skipped.append((stem, "too few points"))
            continue

        wh = img_size(gt_img)
        if wh is None:
            skipped.append((stem, "gt img size fail"))
            continue
        gw, gh = wh

        if pred_size and len(pred_size) >= 2:
            pw, ph = pred_size[0], pred_size[1]
        elif pred_img and os.path.exists(pred_img):
            wh2 = img_size(pred_img)
            if wh2 is None:
                skipped.append((stem, "pred img size fail"))
                continue
            pw, ph = wh2
        else:
            skipped.append((stem, "no pred size"))
            continue

        m = evaluate(gt_arr, pred_arr, gw, gh, pw, ph)
        if m is None:
            skipped.append((stem, "interp fail"))
            continue

        rec = {
            "stem": stem,
            "station": station,
            "gt_image": gt_img,
            "gt_json": gt_json,
            "gt_label": gt_poly.get("label"),
            "gt_n_points": int(len(gt_arr)),
            "pred_image": pred_img,
            "pred_json": pred_json,
            "pred_label": pred_poly.get("label"),
            "pred_n_points": int(len(pred_arr)),
            "orig_image_size": [gw, gh],
            "pred_image_size": [pw, ph],
            **m,
        }
        records.append(rec)

    info = {
        "n_gt": len(gt_map),
        "n_pred": len(pred_map),
        "n_common": len(common),
        "n_evaluated": len(records),
        "skipped": skipped,
        "gt_only": sorted(set(gt_map) - set(pred_map)),
        "pred_only": sorted(set(pred_map) - set(gt_map)),
    }
    return records, info


def main():
    parser = argparse.ArgumentParser(description="评估 VLM 电离图描迹与 GT 精度")
    parser.add_argument("--paper", type=str, default=PAPER,
                        help="Paper 数据根目录 (默认: skill/../Paper)")
    parser.add_argument("--out-json", type=str, default=None,
                        help="输出 JSON 路径 (默认: <paper>/test_eval.json)")
    parser.add_argument("--out-csv", type=str, default=None,
                        help="输出 CSV 路径 (默认: <paper>/test_eval.csv)")
    args = parser.parse_args()

    paper = os.path.abspath(args.paper)
    out_json = args.out_json or os.path.join(paper, "test_eval.json")
    out_csv = args.out_csv or os.path.join(paper, "test_eval.csv")

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
        print(f"[{station}] GT={info['n_gt']}  pred={info['n_pred']}  "
              f"common={info['n_common']}  evaluated={info['n_evaluated']}  "
              f"skip={len(info['skipped'])}")

    # per-station systematic offset correction
    offsets = {}
    for station, _, _ in stations:
        recs = out[station]
        if not recs:
            offsets[station] = {"fof2_mhz": 0.0, "hf2_km": 0.0}
            continue
        fof2_off = sum(r["foF2_err_mhz"] for r in recs) / len(recs)
        hf2_off = sum(r["hF2_err_km"] for r in recs) / len(recs)
        for r in recs:
            r["foF2_err_corrected_mhz"] = r["foF2_err_mhz"] - fof2_off
            r["hF2_err_corrected_km"] = r["hF2_err_km"] - hf2_off
        offsets[station] = {"fof2_mhz": fof2_off, "hf2_km": hf2_off}

    print("\n端点人工系统误差校正 (per-station):")
    for station, off in offsets.items():
        if out.get(station):
            print(f"  [{station}] foF2_offset={off['fof2_mhz']:+.4f}MHz  "
                  f"hF2_offset={off['hf2_km']:+.4f}km")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"stations": out, "info": station_infos,
                   "offsets_per_station": offsets},
                  f, ensure_ascii=False, indent=2)
    print(f"\nJSON: {out_json}")

    cols = [
        "station", "stem",
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

    print("\n=== 汇总 ===")
    for station, _, _ in stations:
        recs = out[station]
        if not recs:
            print(f"  {station:8s} n=0  (无评估记录)")
            continue
        rmse = np.array([r["rmse_km"] for r in recs])
        r2 = np.array([r["pearson_r2"] for r in recs if r["pearson_r2"] == r["pearson_r2"]])
        mean_err = np.array([r["mean_err_km"] for r in recs])
        foF2_err = np.array([r["foF2_err_mhz"] for r in recs])
        foF2_corr = np.array([r["foF2_err_corrected_mhz"] for r in recs])
        foF2_gt = np.array([r["foF2_gt_mhz"] for r in recs])
        foF2_pred = np.array([r["foF2_pred_mhz"] for r in recs])
        foF2_rmse = float(np.sqrt(np.mean(foF2_err ** 2)))
        foF2_rmse_corr = float(np.sqrt(np.mean(foF2_corr ** 2)))
        hF2_err = np.array([r["hF2_err_km"] for r in recs])
        hF2_corr = np.array([r["hF2_err_corrected_km"] for r in recs])
        hF2_gt = np.array([r["hF2_gt_km"] for r in recs])
        hF2_pred = np.array([r["hF2_pred_km"] for r in recs])
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
        print(f"  {station:8s} n={len(recs):3d}  "
              f"rmse={rmse.mean():.2f}km  r2={r2.mean() if len(r2) else float('nan'):.4f}  "
              f"mean_err={mean_err.mean():+.2f}km")
        print(f"           "
              f"foF2_rmse={foF2_rmse:.2f}MHz({foF2_rmse_corr:.2f}corr)  foF2_r2={foF2_r2:.4f}  "
              f"hF2_rmse={hF2_rmse:.2f}km({hF2_rmse_corr:.2f}corr)  hF2_r2={hF2_r2:.4f}")

    if all_records:
        print("\nRMSE 最大的 10 张:")
        for r in sorted(all_records, key=lambda x: -x["rmse_km"])[:10]:
            print(f"  {r['rmse_km']:.2f}km  r2={r['pearson_r2']:.3f}  "
                  f"foF2_err={r['foF2_err_mhz']:+.2f}MHz  "
                  f"[{r['station']}]  {r['stem']}  "
                  f"(gt={r['gt_label']} pred={r['pred_label']})")


if __name__ == "__main__":
    main()
