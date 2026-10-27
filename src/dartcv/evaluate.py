"""Quantitative and visual evaluation harness for dartCV.

Benchmarks the automated board detection and dart localization pipeline
against human-verified ground-truth annotations in data/annotations/ground_truth.json.
Generates comprehensive quantitative metrics (Markdown report + JSON),
diagnostic composite visualizations, and an interactive HTML dashboard in out/eval/.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .annotate import calibrate_from_wire_grid
from .detect import DetectionError, analyze_image, detect_board
from .geometry import RADII, SECTORS, calibration_model_points, compute_homography, polar, project, score_point_detailed
from .overlay import draw_overlay, warp_topdown
from .types import BoardAnalysis, DartDetection

DEFAULT_ANNOTATIONS = Path("data/annotations/ground_truth.json")
DEFAULT_OUT = Path("out/eval")


def match_darts(
    gt_tips_mm: np.ndarray, pred_tips_mm: np.ndarray, max_dist_mm: float = 15.0
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Minimum-weight bipartite matching between GT and predicted darts in board mm."""
    n_gt = len(gt_tips_mm)
    n_pred = len(pred_tips_mm)
    if n_gt == 0 or n_pred == 0:
        return [], list(range(n_gt)), list(range(n_pred))

    cost_matrix = np.zeros((n_gt, n_pred), dtype=np.float64)
    for i in range(n_gt):
        for j in range(n_pred):
            cost_matrix[i, j] = np.linalg.norm(gt_tips_mm[i] - pred_tips_mm[j])

    best_cost = float("inf")
    best_matching: list[tuple[int, int]] = []

    if n_gt <= n_pred:
        for p in itertools.permutations(range(n_pred), n_gt):
            cur_cost = sum(cost_matrix[i, p[i]] for i in range(n_gt))
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_matching = [(i, p[i]) for i in range(n_gt)]
    else:
        for p in itertools.permutations(range(n_gt), n_pred):
            cur_cost = sum(cost_matrix[p[j], j] for j in range(n_pred))
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_matching = [(p[j], j) for j in range(n_pred)]

    matches: list[tuple[int, int, float]] = []
    unmatched_gt = set(range(n_gt))
    unmatched_pred = set(range(n_pred))

    for i, j in best_matching:
        if cost_matrix[i, j] <= max_dist_mm:
            matches.append((i, j, float(cost_matrix[i, j])))
            unmatched_gt.discard(i)
            unmatched_pred.discard(j)

    return matches, sorted(unmatched_gt), sorted(unmatched_pred)


def evaluate_dataset(
    annotations_path: Path,
    out_dir: Path,
    max_error_mm: float = 15.0,
    save_composites: bool = True,
) -> dict:
    """Run full evaluation on ground-truth dataset and generate reports and diagnostics."""
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    with open(annotations_path) as f:
        gt_data = json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)
    images_dict = gt_data.get("images", {})
    image_paths = gt_data.get("starter_15", list(images_dict.keys()))

    total_images = len(image_paths)
    verified_images = 0
    gt_darts_count = 0
    pred_darts_count = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    tip_errors_mm: list[float] = []
    tip_errors_px: list[float] = []
    label_matches = 0
    point_matches = 0
    total_score_matches = 0
    count_matches = 0

    timings_board: list[float] = []
    timings_darts: list[float] = []
    timings_total: list[float] = []

    per_image_results: list[dict] = []
    board_pose_results: list[dict] = []
    all_wg_errors_px: list[float] = []
    all_wg_errors_mm: list[float] = []
    all_center_errors_px: list[float] = []
    all_center_errors_mm: list[float] = []
    all_corner_errors_px: list[float] = []
    all_corner_errors_mm: list[float] = []

    cardinal_angles = [9.0, 99.0, 189.0, 279.0]
    categories_data = {
        "raw_board": {"images": 0, "gt_darts": 0, "pred_darts": 0, "tp": 0, "fp": 0, "fn": 0, "count_match": 0, "score_match": 0, "tip_errors_mm": [], "total_ms": []},
        "single_dart": {"images": 0, "gt_darts": 0, "pred_darts": 0, "tp": 0, "fp": 0, "fn": 0, "count_match": 0, "score_match": 0, "tip_errors_mm": [], "total_ms": []},
        "multi_dart": {"images": 0, "gt_darts": 0, "pred_darts": 0, "tp": 0, "fp": 0, "fn": 0, "count_match": 0, "score_match": 0, "tip_errors_mm": [], "total_ms": []},
    }

    for path_str in image_paths:
        img_path = Path(path_str)
        if not img_path.is_file():
            print(f"Skipping missing image: {img_path}", file=sys.stderr)
            continue

        cat = "raw_board" if "raw_board" in path_str else ("single_dart" if "single_dart" in path_str else "multi_dart")
        categories_data[cat]["images"] += 1

        gt_entry = images_dict.get(path_str, {})
        is_verified = bool(gt_entry.get("verified", False))
        if is_verified:
            verified_images += 1

        gt_darts = gt_entry.get("darts", [])
        gt_total_score = gt_entry.get("total_score", 0)
        n_gt = len(gt_darts)
        gt_darts_count += n_gt
        categories_data[cat]["gt_darts"] += n_gt

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Run pipeline
        try:
            analysis = analyze_image(img, diagnostics=True)
            det_board = analysis.board
            pred_darts = analysis.darts
            pred_total = analysis.total_score
            timings = analysis.timings_ms
        except Exception as exc:
            print(f"Analysis error on {img_path.name}: {exc}", file=sys.stderr)
            det_board = None
            pred_darts = []
            pred_total = 0
            timings = {"board_ms": 0.0, "darts_ms": 0.0, "total_ms": 0.0}

        t_board = timings.get("board_ms", 0.0)
        t_darts = timings.get("darts_ms", 0.0)
        t_total = timings.get("total_ms", 0.0)
        timings_board.append(t_board)
        timings_darts.append(t_darts)
        timings_total.append(t_total)
        categories_data[cat]["total_ms"].append(t_total)

        # Evaluate Board Detection & 24-point Wire Grid Pose Alignment on verified raw_board images
        wg_pts = gt_entry.get("wire_grid_points_px")
        board_metrics_item = None
        if wg_pts and len(wg_pts) >= 4 and det_board is not None:
            H_gt, rings_gt, wires_gt, c_gt, radii_gt = calibrate_from_wire_grid(wg_pts)
            model_24 = []
            for s in range(4):
                for k in range(5):
                    model_24.append(polar(radii_gt[k], cardinal_angles[s]))
            for s in range(4):
                model_24.append(polar(radii_gt[5], cardinal_angles[s]))
            model_24 = np.float32(model_24)

            pts_pred = project(det_board.H, model_24)
            err_px_arr = np.linalg.norm(pts_pred - np.float32(wg_pts), axis=1)
            pts_pred_mm = project(np.linalg.inv(H_gt), pts_pred)
            err_mm_arr = np.linalg.norm(pts_pred_mm - model_24, axis=1)

            wg_err_px = float(np.mean(err_px_arr))
            wg_med_px = float(np.median(err_px_arr))
            wg_err_mm = float(np.mean(err_mm_arr))
            wg_med_mm = float(np.median(err_mm_arr))

            outer_4_indices = [0, 5, 10, 15]
            corner_err_px = float(np.mean(err_px_arr[outer_4_indices]))
            corner_err_mm = float(np.mean(err_mm_arr[outer_4_indices]))

            c_pred = project(det_board.H, [[0, 0]])[0]
            center_err_px = float(np.linalg.norm(np.array(c_gt) - c_pred))
            center_err_mm = float(np.linalg.norm(project(np.linalg.inv(H_gt), [c_pred])[0]))

            all_wg_errors_px.extend(err_px_arr.tolist())
            all_wg_errors_mm.extend(err_mm_arr.tolist())
            all_center_errors_px.append(center_err_px)
            all_center_errors_mm.append(center_err_mm)
            all_corner_errors_px.append(corner_err_px)
            all_corner_errors_mm.append(corner_err_mm)

            board_metrics_item = {
                "image": img_path.name,
                "path": str(img_path),
                "wire_grid_24pt_error_px": round(wg_err_px, 2),
                "wire_grid_24pt_median_px": round(wg_med_px, 2),
                "wire_grid_24pt_error_mm": round(wg_err_mm, 2),
                "wire_grid_24pt_median_mm": round(wg_med_mm, 2),
                "corner_wire_error_px": round(corner_err_px, 2),
                "corner_wire_error_mm": round(corner_err_mm, 2),
                "center_error_px": round(center_err_px, 2),
                "center_error_mm": round(center_err_mm, 2),
                "ecc_score": round(float(det_board.ecc_score), 4) if det_board.ecc_score is not None else 0.0,
            }
            board_pose_results.append(board_metrics_item)

        n_pred = len(pred_darts)
        pred_darts_count += n_pred
        categories_data[cat]["pred_darts"] += n_pred

        if n_gt == n_pred:
            count_matches += 1
            categories_data[cat]["count_match"] += 1

        gt_labels = [d.get("label", "") for d in gt_darts]
        pred_labels = [d.label for d in pred_darts]

        if n_gt == 0 and n_pred == 0:
            total_score_matches += 1
            categories_data[cat]["score_match"] += 1
            status = "PASS"
            matches = []
            img_errors_mm = []
            img_errors_px = []
        else:
            H = det_board.H if det_board is not None else np.eye(3)
            H_inv = np.linalg.inv(H)

            gt_tips_px = np.array([d["tip_px"] for d in gt_darts], dtype=np.float32).reshape(-1, 2)
            if len(gt_tips_px) > 0:
                gt_tips_mm = project(H_inv, gt_tips_px)
            else:
                gt_tips_mm = np.empty((0, 2), dtype=np.float32)

            pred_tips_mm = np.array([d.tip_mm for d in pred_darts], dtype=np.float32).reshape(-1, 2)
            pred_tips_px = np.array([d.tip_px for d in pred_darts], dtype=np.float32).reshape(-1, 2)

            matches, un_gt, un_pred = match_darts(gt_tips_mm, pred_tips_mm, max_dist_mm=max_error_mm)

            tp = len(matches)
            fp = len(un_pred)
            fn = len(un_gt)
            true_positives += tp
            false_positives += fp
            false_negatives += fn
            categories_data[cat]["tp"] += tp
            categories_data[cat]["fp"] += fp
            categories_data[cat]["fn"] += fn

            img_errors_mm = []
            img_errors_px = []
            for i_gt, j_pred, dist_mm in matches:
                tip_errors_mm.append(dist_mm)
                img_errors_mm.append(dist_mm)
                categories_data[cat]["tip_errors_mm"].append(dist_mm)

                px_dist = float(np.linalg.norm(gt_tips_px[i_gt] - pred_tips_px[j_pred]))
                tip_errors_px.append(px_dist)
                img_errors_px.append(px_dist)

                gt_lbl = gt_darts[i_gt].get("label", "")
                pred_lbl = pred_darts[j_pred].label
                if gt_lbl == pred_lbl:
                    label_matches += 1

                gt_sc = gt_darts[i_gt].get("score", 0)
                pred_sc = pred_darts[j_pred].score
                if gt_sc == pred_sc:
                    point_matches += 1

            if pred_total == gt_total_score:
                total_score_matches += 1
                categories_data[cat]["score_match"] += 1

            status = "PASS" if (fn == 0 and fp == 0 and pred_total == gt_total_score) else "FAIL"

        composite_filename = f"{img_path.stem}_eval.png"
        per_image_results.append({
            "image": img_path.name,
            "path": str(img_path),
            "category": cat,
            "verified": is_verified,
            "gt_darts": n_gt,
            "pred_darts": n_pred,
            "gt_labels": gt_labels,
            "pred_labels": pred_labels,
            "gt_score": gt_total_score,
            "pred_score": pred_total,
            "mean_error_mm": float(np.mean(img_errors_mm)) if img_errors_mm else 0.0,
            "mean_error_px": float(np.mean(img_errors_px)) if img_errors_px else 0.0,
            "board_ms": round(t_board, 1),
            "darts_ms": round(t_darts, 1),
            "total_ms": round(t_total, 1),
            "status": status,
            "composite_file": composite_filename,
            "board_alignment": board_metrics_item,
            "gt_darts_detail": gt_darts,
            "pred_darts_detail": [
                {
                    "tip_px": [float(d.tip_px[0]), float(d.tip_px[1])],
                    "tail_px": [float(d.tail_px[0]), float(d.tail_px[1])] if d.tail_px is not None else None,
                    "label": str(d.label),
                    "score": int(d.score),
                    "confidence": round(float(d.confidence), 2),
                }
                for d in pred_darts
            ],
        })

        if save_composites and det_board is not None:
            _write_composite(img, det_board, gt_darts, pred_darts, out_dir / composite_filename, status)

    # Compute aggregate metrics
    p_denom = true_positives + false_positives
    r_denom = true_positives + false_negatives
    precision = (true_positives / p_denom * 100.0) if p_denom > 0 else 0.0
    recall = (true_positives / r_denom * 100.0) if r_denom > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    count_acc = (count_matches / total_images * 100.0) if total_images > 0 else 0.0
    total_score_acc = (total_score_matches / total_images * 100.0) if total_images > 0 else 0.0
    label_acc = (label_matches / true_positives * 100.0) if true_positives > 0 else 0.0
    point_acc = (point_matches / true_positives * 100.0) if true_positives > 0 else 0.0

    # Category summaries
    cat_summary = {}
    for c_name, c_dict in categories_data.items():
        c_p_den = c_dict["tp"] + c_dict["fp"]
        c_r_den = c_dict["tp"] + c_dict["fn"]
        c_prec = (c_dict["tp"] / c_p_den * 100.0) if c_p_den > 0 else 0.0
        c_rec = (c_dict["tp"] / c_r_den * 100.0) if c_r_den > 0 else 0.0
        c_f1 = (2 * c_prec * c_rec / (c_prec + c_rec)) if (c_prec + c_rec) > 0 else 0.0
        cat_summary[c_name] = {
            "images": c_dict["images"],
            "gt_darts": c_dict["gt_darts"],
            "pred_darts": c_dict["pred_darts"],
            "tp": c_dict["tp"],
            "fp": c_dict["fp"],
            "fn": c_dict["fn"],
            "precision_pct": round(c_prec, 1),
            "recall_pct": round(c_rec, 1),
            "f1_score_pct": round(c_f1, 1),
            "count_accuracy_pct": round(c_dict["count_match"] / c_dict["images"] * 100.0, 1) if c_dict["images"] > 0 else 0.0,
            "turn_score_accuracy_pct": round(c_dict["score_match"] / c_dict["images"] * 100.0, 1) if c_dict["images"] > 0 else 0.0,
            "mean_tip_error_mm": round(float(np.mean(c_dict["tip_errors_mm"])), 2) if c_dict["tip_errors_mm"] else 0.0,
            "mean_total_ms": round(float(np.mean(c_dict["total_ms"])), 1) if c_dict["total_ms"] else 0.0,
        }

    metrics = {
        "dataset_summary": {
            "total_images": total_images,
            "verified_images": verified_images,
            "gt_darts": gt_darts_count,
            "predicted_darts": pred_darts_count,
        },
        "detection_metrics": {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision_pct": round(precision, 2),
            "recall_pct": round(recall, 2),
            "f1_score_pct": round(f1, 2),
            "count_accuracy_pct": round(count_acc, 2),
        },
        "board_pose_accuracy": {
            "annotated_boards": len(board_pose_results),
            "mean_24pt_wire_error_px": round(float(np.mean(all_wg_errors_px)), 2) if all_wg_errors_px else 0.0,
            "median_24pt_wire_error_px": round(float(np.median(all_wg_errors_px)), 2) if all_wg_errors_px else 0.0,
            "mean_24pt_wire_error_mm": round(float(np.mean(all_wg_errors_mm)), 2) if all_wg_errors_mm else 0.0,
            "median_24pt_wire_error_mm": round(float(np.median(all_wg_errors_mm)), 2) if all_wg_errors_mm else 0.0,
            "mean_center_error_px": round(float(np.mean(all_center_errors_px)), 2) if all_center_errors_px else 0.0,
            "mean_center_error_mm": round(float(np.mean(all_center_errors_mm)), 2) if all_center_errors_mm else 0.0,
            "mean_corner_wire_error_px": round(float(np.mean(all_corner_errors_px)), 2) if all_corner_errors_px else 0.0,
            "mean_corner_wire_error_mm": round(float(np.mean(all_corner_errors_mm)), 2) if all_corner_errors_mm else 0.0,
            "per_board": board_pose_results,
        },
        "tip_localization_error": {
            "median_mm": round(float(np.median(tip_errors_mm)), 2) if tip_errors_mm else 0.0,
            "mean_mm": round(float(np.mean(tip_errors_mm)), 2) if tip_errors_mm else 0.0,
            "p90_mm": round(float(np.percentile(tip_errors_mm, 90)), 2) if tip_errors_mm else 0.0,
            "median_px": round(float(np.median(tip_errors_px)), 2) if tip_errors_px else 0.0,
            "mean_px": round(float(np.mean(tip_errors_px)), 2) if tip_errors_px else 0.0,
            "p90_px": round(float(np.percentile(tip_errors_px, 90)), 2) if tip_errors_px else 0.0,
        },
        "scoring_accuracy": {
            "label_accuracy_pct": round(label_acc, 2),
            "point_accuracy_pct": round(point_acc, 2),
            "turn_total_score_accuracy_pct": round(total_score_acc, 2),
        },
        "runtime_latency_ms": {
            "board_mean": round(float(np.mean(timings_board)), 1) if timings_board else 0.0,
            "board_median": round(float(np.median(timings_board)), 1) if timings_board else 0.0,
            "board_p95": round(float(np.percentile(timings_board, 95)), 1) if timings_board else 0.0,
            "darts_mean": round(float(np.mean(timings_darts)), 1) if timings_darts else 0.0,
            "darts_median": round(float(np.median(timings_darts)), 1) if timings_darts else 0.0,
            "darts_p95": round(float(np.percentile(timings_darts, 95)), 1) if timings_darts else 0.0,
            "total_mean": round(float(np.mean(timings_total)), 1) if timings_total else 0.0,
            "total_median": round(float(np.median(timings_total)), 1) if timings_total else 0.0,
            "total_p95": round(float(np.percentile(timings_total, 95)), 1) if timings_total else 0.0,
        },
        "category_breakdown": cat_summary,
        "per_image": per_image_results,
    }

    # Save metrics.json
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Generate Markdown Report
    report_md = _generate_markdown_report(metrics)
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")

    return metrics


def _generate_markdown_report(m: dict) -> str:
    lines = [
        "# DartCV Quantitative Evaluation Report",
        "",
        "## 1. Benchmark Summary",
        f"- **Total Images:** {m['dataset_summary']['total_images']} ({m['dataset_summary']['verified_images']} verified)",
        f"- **Ground-Truth Darts:** {m['dataset_summary']['gt_darts']}",
        f"- **Predicted Darts:** {m['dataset_summary']['predicted_darts']}",
        "",
        "## 2. Detection Performance",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **Dart Count Accuracy** | {m['detection_metrics']['count_accuracy_pct']}% |",
        f"| **Precision** | {m['detection_metrics']['precision_pct']}% (TP: {m['detection_metrics']['true_positives']}, FP: {m['detection_metrics']['false_positives']}) |",
        f"| **Recall** | {m['detection_metrics']['recall_pct']}% (FN: {m['detection_metrics']['false_negatives']}) |",
        f"| **F1 Score** | {m['detection_metrics']['f1_score_pct']}% |",
        "",
        "## 3. Board Pose & Alignment Accuracy (Ground-Truth 24-Point Wire Grid)",
        f"- **Annotated Boards:** {m['board_pose_accuracy']['annotated_boards']} verified raw board images",
        f"- **Mean 24-Point Wire Grid Error:** {m['board_pose_accuracy']['mean_24pt_wire_error_px']} px ({m['board_pose_accuracy']['mean_24pt_wire_error_mm']} mm)",
        f"- **Median 24-Point Wire Grid Error:** {m['board_pose_accuracy']['median_24pt_wire_error_px']} px ({m['board_pose_accuracy']['median_24pt_wire_error_mm']} mm)",
        f"- **Mean Board Center Error:** {m['board_pose_accuracy']['mean_center_error_px']} px ({m['board_pose_accuracy']['mean_center_error_mm']} mm)",
        f"- **Mean Outer-Double Corner Error:** {m['board_pose_accuracy']['mean_corner_wire_error_px']} px ({m['board_pose_accuracy']['mean_corner_wire_error_mm']} mm)",
        "",
        "| Image | 24-Pt Error (px) | 24-Pt Error (mm) | Center Error (px / mm) | ECC Score |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for b in m["board_pose_accuracy"].get("per_board", []):
        lines.append(
            f"| `{b['image']}` | {b['wire_grid_24pt_error_px']} px | {b['wire_grid_24pt_error_mm']} mm | {b['center_error_px']} px ({b['center_error_mm']} mm) | {b['ecc_score']} |"
        )

    lines.extend([
        "",
        "## 4. Tip Localization Error (Matched Darts)",
        "| Error Metric | Board Distance (mm) | Original Image (px) |",
        "| :--- | :--- | :--- |",
        f"| **Median** | {m['tip_localization_error']['median_mm']} mm | {m['tip_localization_error']['median_px']} px |",
        f"| **Mean** | {m['tip_localization_error']['mean_mm']} mm | {m['tip_localization_error']['mean_px']} px |",
        f"| **90th Percentile (P90)** | {m['tip_localization_error']['p90_mm']} mm | {m['tip_localization_error']['p90_px']} px |",
        "",
        "## 5. Scoring Accuracy",
        "| Metric | Accuracy |",
        "| :--- | :--- |",
        f"| **Exact Sector/Multiplier Label** | {m['scoring_accuracy']['label_accuracy_pct']}% |",
        f"| **Per-Dart Points** | {m['scoring_accuracy']['point_accuracy_pct']}% |",
        f"| **Turn Total Score Exact Match** | {m['scoring_accuracy']['turn_total_score_accuracy_pct']}% |",
        "",
        "## 6. Category Breakdown",
        "| Category | Images | GT Darts | Pred Darts | Precision | Recall | Count Acc | Score Acc | Mean Tip Error | Latency |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for c_name, c in m.get("category_breakdown", {}).items():
        lines.append(
            f"| **{c_name}** | {c['images']} | {c['gt_darts']} | {c['pred_darts']} | {c['precision_pct']}% | {c['recall_pct']}% | {c['count_accuracy_pct']}% | {c['turn_score_accuracy_pct']}% | {c['mean_tip_error_mm']} mm | {c['mean_total_ms']} ms |"
        )

    lines.extend([
        "",
        "## 7. Runtime Latency (ms)",
        "| Pipeline Stage | Mean | Median | 95th Percentile (P95) |",
        "| :--- | :--- | :--- | :--- |",
        f"| Board Alignment (ECC) | {m['runtime_latency_ms']['board_mean']} ms | {m['runtime_latency_ms']['board_median']} ms | {m['runtime_latency_ms']['board_p95']} ms |",
        f"| Dart Detection | {m['runtime_latency_ms']['darts_mean']} ms | {m['runtime_latency_ms']['darts_median']} ms | {m['runtime_latency_ms']['darts_p95']} ms |",
        f"| **Total End-to-End** | **{m['runtime_latency_ms']['total_mean']} ms** | **{m['runtime_latency_ms']['total_median']} ms** | **{m['runtime_latency_ms']['total_p95']} ms** |",
        "",
        "## 8. Per-Image Breakdown",
        "| Image | Category | GT Darts | Pred Darts | GT Score | Pred Score | Mean Error (mm) | Total Latency (ms) | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for item in m["per_image"]:
        lines.append(
            f"| `{item['image']}` | {item.get('category', '')} | {item['gt_darts']} | {item['pred_darts']} | {item['gt_score']} | {item['pred_score']} | {item['mean_error_mm']:.1f} | {item['total_ms']:.1f} | **{item['status']}** |"
        )

    return "\n".join(lines) + "\n"


def _write_composite(
    img: np.ndarray,
    det_board,
    gt_darts: list[dict],
    pred_darts: list[DartDetection],
    out_path: Path,
    status: str,
) -> None:
    """Write side-by-side diagnostic visualization: original perspective vs topdown."""
    H = det_board.H
    h, w = img.shape[:2]
    scale = 1000.0 / h
    disp_w = int(w * scale)
    disp_h = 1000

    left = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    # Scale homography to display size
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float32)
    H_disp = S @ H
    left = draw_overlay(left, H_disp)

    # Draw GT tips (magenta circles)
    for idx, d in enumerate(gt_darts):
        pt_disp = (int(d["tip_px"][0] * scale), int(d["tip_px"][1] * scale))
        cv2.circle(left, pt_disp, 8, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(left, pt_disp, 12, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            left,
            f"GT:{d.get('label', '')}",
            (pt_disp[0] + 14, pt_disp[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # Draw Pred tips (green crosshairs)
    for idx, d in enumerate(pred_darts):
        pt_disp = (int(d.tip_px[0] * scale), int(d.tip_px[1] * scale))
        cv2.drawMarker(left, pt_disp, (0, 255, 0), cv2.MARKER_TILTED_CROSS, 20, 2, cv2.LINE_AA)
        cv2.putText(
            left,
            f"Pred:{d.label}",
            (pt_disp[0] + 14, pt_disp[1] + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # Status badge
    colour = (0, 200, 0) if status == "PASS" else (0, 0, 220)
    cv2.rectangle(left, (10, 10), (140, 50), colour, -1)
    cv2.putText(left, status, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # Right panel: topdown view
    topdown = warp_topdown(img, H)
    topdown = cv2.resize(topdown, (disp_h, disp_h), interpolation=cv2.INTER_AREA)

    composite = np.hstack([left, topdown])
    cv2.imwrite(str(out_path), composite)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="dartcv-eval", description="Evaluate dartCV against ground truth")
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS), help="Ground truth JSON file")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory for report and diagnostics")
    parser.add_argument("--max-error-mm", type=float, default=15.0, help="Matching tolerance gate in mm (default: 15.0)")
    parser.add_argument("--no-composites", action="store_true", help="Skip generating visual composite images")
    args = parser.parse_args(argv)

    ann_path = Path(args.annotations)
    out_dir = Path(args.out)

    print(f"Running evaluation against {ann_path}...")
    metrics = evaluate_dataset(
        ann_path,
        out_dir,
        max_error_mm=args.max_error_mm,
        save_composites=not args.no_composites,
    )

    report_text = _generate_markdown_report(metrics)
    print("\n" + report_text)
    print(f"Wrote evaluation report to {out_dir / 'report.md'}")
    print(f"Wrote metrics JSON to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
