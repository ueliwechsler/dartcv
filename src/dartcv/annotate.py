"""Interactive annotation & evaluation dashboard tool for dartcv ground-truth dataset.

Provides:
1. A rich Web UI with:
   - Pipeline Evaluation & Benchmark Telemetry Dashboard (Section A KPIs, Section B 15-image benchmark table + inspector).
   - Fresh / Unseen Images Explorer (Section C live analysis on 63 unannotated images + 9-image gallery).
   - Ground-Truth Annotator with pan/zoom (including right-mouse pan), 4x magnifier loupe, 24-point Joint Projective ALS wire grid calibration, and real-time score calculation.
2. A desktop OpenCV window interface when a display server is available.
"""

from __future__ import annotations

import argparse
import http.server
import json
import math
import os
import socketserver
import sys
import threading
import time
import urllib.parse
from pathlib import Path

import cv2
import numpy as np

from .detect import DetectionError, _anchor_bull, analyze_image, detect_board
from .geometry import (
    CALIB_ANGLES,
    CALIB_NAMES,
    RADII,
    SECTORS,
    calibration_model_points,
    compute_homography,
    polar,
    project,
    score_point_detailed,
)
from .overlay import draw_overlay, warp_topdown
from .types import BoardAnalysis, DartDetection

DEFAULT_ANNOTATIONS = Path("data/annotations/ground_truth.json")
DEFAULT_IMAGES_DIR = Path("data/images")
DEFAULT_OUT = Path("out/eval")

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>DartCV Telemetry & Annotation Studio</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {
      --bg: #0b1120;
      --card: #1e293b;
      --card-border: #334155;
      --card-hover: #273549;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --accent-hover: #0284c7;
      --success: #22c55e;
      --warning: #f59e0b;
      --danger: #ef4444;
      --purple: #a855f7;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }
    header {
      background: var(--card);
      border-bottom: 1px solid var(--card-border);
      padding: 10px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      flex-shrink: 0;
      z-index: 40;
    }
    .header-left, .header-right {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    h1 {
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: var(--accent);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .nav-tabs {
      display: flex;
      gap: 4px;
      background: #070d19;
      padding: 3px;
      border-radius: 8px;
      border: 1px solid var(--card-border);
      margin-left: 8px;
    }
    .nav-tab {
      padding: 6px 14px;
      font-size: 0.85rem;
      font-weight: 600;
      border-radius: 6px;
      border: none;
      background: transparent;
      color: var(--text-muted);
      cursor: pointer;
      transition: all 0.15s;
    }
    .nav-tab:hover {
      color: var(--text);
      background: rgba(255,255,255,0.05);
    }
    .nav-tab.active {
      background: #0284c7;
      color: #ffffff;
      box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    select, button, input {
      font-size: 0.875rem;
      border-radius: 6px;
      border: 1px solid var(--card-border);
      background: #070d19;
      color: var(--text);
      padding: 6px 12px;
      outline: none;
    }
    select { cursor: pointer; max-width: 320px; }
    button {
      cursor: pointer;
      background: #334155;
      border: none;
      font-weight: 500;
      transition: background 0.15s;
    }
    button:hover { background: #475569; }
    button.btn-primary { background: var(--accent); color: #04121f; font-weight: 600; }
    button.btn-primary:hover { background: var(--accent-hover); }
    button.btn-success { background: var(--success); color: #052e16; font-weight: 600; }
    button.btn-danger { background: var(--danger); color: #fff; }
    .badge {
      font-size: 0.75rem;
      font-weight: 700;
      padding: 3px 8px;
      border-radius: 9999px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .badge-verified { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid #22c55e; }
    .badge-unverified { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
    .badge-pass { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid #22c55e; }
    .badge-fail { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }

    /* Main View Panels */
    .view-panel {
      flex: 1;
      display: flex;
      overflow: hidden;
      position: relative;
    }

    /* Dashboard Layout */
    #dashboard-view {
      overflow-y: auto;
      padding: 24px;
      flex-direction: column;
      gap: 24px;
      background: #090d16;
    }
    .dash-hero {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--card);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 20px 24px;
      gap: 16px;
    }
    .dash-hero-title h2 {
      font-size: 1.4rem;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .dash-hero-title p {
      font-size: 0.875rem;
      color: var(--text-muted);
      margin-top: 4px;
    }
    .dash-hero-actions {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    /* KPI Grid */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 16px;
    }
    .kpi-card {
      background: var(--card);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      transition: transform 0.15s, border-color 0.15s;
    }
    .kpi-card:hover {
      transform: translateY(-2px);
      border-color: #475569;
    }
    .kpi-header {
      font-size: 0.8rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .kpi-value {
      font-size: 1.85rem;
      font-weight: 800;
      color: #fff;
      line-height: 1.1;
    }
    .kpi-sub {
      font-size: 0.8rem;
      color: var(--text-muted);
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .kpi-bar {
      height: 8px;
      background: #070d19;
      border-radius: 4px;
      overflow: hidden;
      display: flex;
      margin-top: 4px;
    }
    .kpi-bar-fill-blue { background: var(--accent); }
    .kpi-bar-fill-orange { background: var(--warning); }
    .kpi-bar-fill-green { background: var(--success); }

    /* Dashboard Sections */
    .section-box {
      background: var(--card);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .section-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }
    .section-header h3 {
      font-size: 1.15rem;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .filter-pills {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }
    .filter-pill {
      font-size: 0.75rem;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 6px;
      border: 1px solid var(--card-border);
      background: #070d19;
      color: var(--text-muted);
      cursor: pointer;
    }
    .filter-pill:hover, .filter-pill.active {
      background: #0284c7;
      color: #fff;
      border-color: #0284c7;
    }

    /* Benchmark Table */
    .table-container {
      overflow-x: auto;
      border-radius: 8px;
      border: 1px solid var(--card-border);
    }
    table.bench-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
      text-align: left;
    }
    table.bench-table th {
      background: #070d19;
      color: var(--text-muted);
      font-weight: 600;
      padding: 10px 14px;
      border-bottom: 1px solid var(--card-border);
    }
    table.bench-table td {
      padding: 10px 14px;
      border-bottom: 1px solid rgba(51, 65, 85, 0.5);
      color: var(--text);
    }
    table.bench-table tr:hover {
      background: var(--card-hover);
      cursor: pointer;
    }

    /* Unseen Gallery Grid */
    .gallery-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(310px, 1fr));
      gap: 16px;
    }
    .gallery-card {
      background: #070d19;
      border: 1px solid var(--card-border);
      border-radius: 10px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      transition: transform 0.15s, border-color 0.15s;
    }
    .gallery-card:hover {
      transform: translateY(-2px);
      border-color: var(--accent);
    }
    .gallery-thumb {
      height: 180px;
      background: #020617;
      position: relative;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .gallery-thumb img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
    .gallery-info {
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      flex: 1;
    }
    .gallery-title {
      font-size: 0.85rem;
      font-weight: 600;
      color: #fff;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .gallery-results {
      font-size: 0.8rem;
      color: var(--text-muted);
      background: rgba(255,255,255,0.03);
      padding: 8px;
      border-radius: 6px;
      border: 1px solid rgba(255,255,255,0.05);
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .gallery-actions {
      display: flex;
      gap: 8px;
      margin-top: auto;
      padding-top: 8px;
    }

    /* Annotator View Styles */
    #annotator-view {
      display: flex;
    }
    #viewport {
      flex: 1;
      background: #020617;
      position: relative;
      overflow: hidden;
      cursor: crosshair;
    }
    #canvas-container {
      position: absolute;
      transform-origin: 0 0;
      will-change: transform;
    }
    #image-canvas { display: block; }
    #overlay-canvas {
      position: absolute;
      top: 0;
      left: 0;
      pointer-events: none;
    }
    #loupe {
      position: absolute;
      top: 16px;
      right: 16px;
      width: 180px;
      height: 180px;
      border-radius: 12px;
      border: 2px solid var(--accent);
      background: #000;
      box-shadow: 0 10px 25px -5px rgba(0,0,0,0.8);
      overflow: hidden;
      pointer-events: none;
      display: none;
      z-index: 50;
    }
    #loupe canvas {
      width: 100%;
      height: 100%;
      image-rendering: pixelated;
    }
    #loupe-coord {
      position: absolute;
      bottom: 4px;
      left: 0;
      right: 0;
      text-align: center;
      font-size: 11px;
      font-family: monospace;
      color: #38bdf8;
      background: rgba(0,0,0,0.6);
      padding: 2px 0;
    }
    #sidebar {
      width: 420px;
      background: var(--card);
      border-left: 1px solid var(--card-border);
      display: flex;
      flex-direction: column;
      flex-shrink: 0;
      overflow-y: auto;
      padding: 16px;
      gap: 16px;
    }
    .panel-section {
      background: #0b1120;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .panel-section h3 {
      font-size: 0.9rem;
      font-weight: 600;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .dart-card {
      background: var(--card);
      border: 1px solid var(--card-border);
      border-radius: 6px;
      padding: 8px 10px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .dart-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-weight: 600;
      font-size: 0.85rem;
    }
    .coord-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      font-size: 0.75rem;
    }
    .coord-item {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .shortcuts {
      font-size: 0.75rem;
      color: var(--text-muted);
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    kbd {
      background: #000;
      padding: 1px 5px;
      border-radius: 4px;
      border: 1px solid #475569;
      font-family: monospace;
      color: #38bdf8;
    }
    .toast {
      position: fixed;
      bottom: 24px;
      left: 50%;
      transform: translateX(-50%);
      background: #0284c7;
      color: #fff;
      padding: 8px 16px;
      border-radius: 8px;
      font-size: 0.875rem;
      font-weight: 600;
      box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5);
      z-index: 100;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.2s;
    }
    .toast.show { opacity: 1; }

    /* Modal */
    .modal-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.85);
      z-index: 1000;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }
    .modal-content {
      background: var(--card);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      max-width: 1400px;
      width: 100%;
      max-height: 92vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      box-shadow: 0 25px 50px -12px rgba(0,0,0,0.8);
    }
    .modal-header {
      padding: 16px 20px;
      border-bottom: 1px solid var(--card-border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .modal-body {
      padding: 20px;
      overflow-y: auto;
      display: flex;
      gap: 20px;
    }
    .modal-image-col {
      flex: 1.5;
      background: #020617;
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .modal-image-col img {
      max-width: 100%;
      max-height: 72vh;
      object-fit: contain;
    }
    .modal-details-col {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 16px;
      overflow-y: auto;
    }
  </style>
</head>
<body>
  <header>
    <div class="header-left">
      <h1>
        <span>🎯 DartCV Studio</span>
      </h1>
      <div class="nav-tabs">
        <button id="tab-btn-dashboard" class="nav-tab active">📊 Pipeline Dashboard</button>
        <button id="tab-btn-annotator" class="nav-tab">🎯 Annotator</button>
      </div>
    </div>

    <!-- Dashboard Header Controls -->
    <div id="dash-header-controls" class="header-right">
      <span id="dash-last-updated" style="font-size:0.75rem; color:var(--text-muted);">15 Benchmark Images</span>
      <button id="btn-rerun-eval" class="btn-primary" style="display:flex; align-items:center; gap:6px;">
        <span>🔄 Re-run Evaluation</span>
      </button>
    </div>

    <!-- Annotator Header Controls -->
    <div id="annotator-header-controls" class="header-right" style="display:none;">
      <select id="image-select"></select>
      <span id="image-progress" style="font-size:0.8rem; color:var(--text-muted); font-family:monospace;"></span>
      <span id="verified-badge" class="badge badge-unverified">Unverified</span>
      <button id="btn-prev" title="Previous Image (P)">&larr; Prev</button>
      <button id="btn-next" title="Next Image (N)">Next &rarr;</button>
      <button id="btn-save" class="btn-success" title="Save & Next (Space / Enter)">💾 Save & Next</button>
    </div>
  </header>

  <!-- TAB 1: DASHBOARD VIEW -->
  <div id="dashboard-view" class="view-panel">
    <div class="dash-hero">
      <div class="dash-hero-title">
        <h2>📊 Pipeline Evaluation & Telemetry Dashboard</h2>
        <p>Quantitative benchmarks against verified ground-truth annotations + real-time generalization testing on unseen test images.</p>
      </div>
      <div class="dash-hero-actions">
        <span class="badge" style="background:#0284c7; color:#fff;">Starter-15 Verified Set</span>
        <span class="badge" style="background:#334155; color:#94a3b8;">63 Unseen Test Set</span>
      </div>
    </div>

    <!-- SECTION A: KPI CARDS -->
    <div class="kpi-grid">
      <!-- Card 1: Detection -->
      <div class="kpi-card">
        <div class="kpi-header">
          <span>Dart Detection</span>
          <span style="color:#38bdf8;">F1 Score</span>
        </div>
        <div id="kpi-f1" class="kpi-value">--%</div>
        <div class="kpi-sub">
          <div style="display:flex; justify-content:space-between;">
            <span>Precision: <strong id="kpi-prec" style="color:#fff;">--%</strong></span>
            <span>Recall: <strong id="kpi-rec" style="color:#fff;">--%</strong></span>
          </div>
          <div>Exact Count Match: <strong id="kpi-count-acc" style="color:#38bdf8;">--%</strong></div>
        </div>
        <div class="kpi-bar">
          <div id="kpi-f1-bar" class="kpi-bar-fill-blue" style="width:0%;"></div>
        </div>
      </div>

      <!-- Card 2: Scoring -->
      <div class="kpi-card">
        <div class="kpi-header">
          <span>Scoring Accuracy</span>
          <span style="color:#22c55e;">BDO Rules</span>
        </div>
        <div id="kpi-label-acc" class="kpi-value">--%</div>
        <div class="kpi-sub">
          <div style="display:flex; justify-content:space-between;">
            <span>Exact Sector: <strong id="kpi-sub-label" style="color:#fff;">--%</strong></span>
            <span>Points: <strong id="kpi-pts-acc" style="color:#fff;">--%</strong></span>
          </div>
          <div>Turn Total Score Match: <strong id="kpi-turn-acc" style="color:#22c55e;">--%</strong></div>
        </div>
        <div class="kpi-bar">
          <div id="kpi-score-bar" class="kpi-bar-fill-green" style="width:0%;"></div>
        </div>
      </div>

      <!-- Card 3: Tip Error -->
      <div class="kpi-card">
        <div class="kpi-header">
          <span>Tip Localization</span>
          <span style="color:#f59e0b;">Board Distance</span>
        </div>
        <div id="kpi-tip-med" class="kpi-value">-- mm</div>
        <div class="kpi-sub">
          <div style="display:flex; justify-content:space-between;">
            <span>Mean: <strong id="kpi-tip-mean" style="color:#fff;">-- mm</strong></span>
            <span>P90: <strong id="kpi-tip-p90" style="color:#fff;">-- mm</strong></span>
          </div>
          <div>Pixel Error: <strong id="kpi-tip-px" style="color:#f59e0b;">-- px</strong></div>
        </div>
        <div class="kpi-bar">
          <div id="kpi-tip-bar" class="kpi-bar-fill-orange" style="width:50%;"></div>
        </div>
      </div>

      <!-- Card 4: Board Pose (24-Pt Grid) -->
      <div class="kpi-card">
        <div class="kpi-header">
          <span>Board Alignment</span>
          <span style="color:#a855f7;">24-Pt Wire Grid</span>
        </div>
        <div id="kpi-wire-err" class="kpi-value">-- mm</div>
        <div class="kpi-sub">
          <div style="display:flex; justify-content:space-between;">
            <span>24-Pt px: <strong id="kpi-wire-px" style="color:#fff;">-- px</strong></span>
            <span>Corner: <strong id="kpi-corner-err" style="color:#fff;">-- mm</strong></span>
          </div>
          <div>Bull Center Error: <strong id="kpi-center-err" style="color:#a855f7;">-- px (-- mm)</strong></div>
        </div>
        <div class="kpi-bar">
          <div class="kpi-bar-fill-blue" style="width:90%; background:#a855f7;"></div>
        </div>
      </div>

      <!-- Card 5: Latency Breakdown -->
      <div class="kpi-card">
        <div class="kpi-header">
          <span>Stage Runtime</span>
          <span style="color:#38bdf8;">End-to-End</span>
        </div>
        <div id="kpi-total-lat" class="kpi-value">-- ms</div>
        <div class="kpi-sub">
          <div style="display:flex; justify-content:space-between;">
            <span style="color:#38bdf8;">Board: <strong id="kpi-board-ms">-- ms</strong></span>
            <span style="color:#f59e0b;">Darts: <strong id="kpi-darts-ms">-- ms</strong></span>
          </div>
          <div>P95 Latency: <strong id="kpi-p95-lat" style="color:#fff;">-- ms</strong></div>
        </div>
        <div class="kpi-bar">
          <div id="kpi-board-lat-bar" class="kpi-bar-fill-blue" style="width:45%;" title="Board Stage"></div>
          <div id="kpi-darts-lat-bar" class="kpi-bar-fill-orange" style="width:55%;" title="Darts Stage"></div>
        </div>
      </div>
    </div>

    <!-- SECTION B: 15-IMAGE BENCHMARK RESULTS TABLE -->
    <div class="section-box">
      <div class="section-header">
        <h3>
          <span>📋 15-Image Verified Benchmark Suite</span>
          <span id="bench-count-badge" class="badge" style="background:#334155; color:#94a3b8;">15 Images</span>
        </h3>
        <div class="filter-pills">
          <button class="filter-pill active" onclick="filterBenchmark('all')">All (15)</button>
          <button class="filter-pill" onclick="filterBenchmark('raw_board')">Raw Board (5)</button>
          <button class="filter-pill" onclick="filterBenchmark('single_dart')">Single Dart (5)</button>
          <button class="filter-pill" onclick="filterBenchmark('multi_dart')">Multi Dart (5)</button>
          <button class="filter-pill" onclick="filterBenchmark('pass')" style="color:#4ade80;">Pass Only</button>
          <button class="filter-pill" onclick="filterBenchmark('fail')" style="color:#f87171;">Fail Only</button>
        </div>
      </div>

      <div class="table-container">
        <table class="bench-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Image</th>
              <th>Category</th>
              <th>GT vs Pred Darts</th>
              <th>GT vs Pred Score</th>
              <th>Tip Error (mm)</th>
              <th>Board 24-Pt (mm)</th>
              <th>Latency (ms)</th>
              <th>Status</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody id="bench-table-body">
            <tr><td colspan="10" style="text-align:center; padding:20px; color:var(--text-muted);">Loading benchmark results...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- SECTION C: UNSEEN IMAGES EXPLORER -->
    <div class="section-box">
      <div class="section-header">
        <div>
          <h3>
            <span>🔭 Fresh / Unseen Images Explorer</span>
            <span id="unseen-count-badge" class="badge" style="background:#0284c7; color:#fff;">63 Images</span>
          </h3>
          <p style="font-size:0.8rem; color:var(--text-muted); margin-top:2px;">
            Test pipeline generalization on unannotated images outside the training set. Run CV detection live on any image.
          </p>
        </div>
        <div style="display:flex; gap:8px; align-items:center;">
          <select id="unseen-select" style="max-width:280px;" onchange="analyzeSelectedUnseen(this.value)">
            <option value="">-- Choose from 63 Unseen Images --</option>
          </select>
        </div>
      </div>

      <!-- 9 Starter Unseen Gallery -->
      <div id="unseen-gallery" class="gallery-grid">
        <!-- Rendered by JS -->
      </div>
    </div>
  </div>

  <!-- TAB 2: ANNOTATOR VIEW -->
  <main id="annotator-view" class="view-panel" style="display:none;">
    <div id="viewport">
      <div id="canvas-container">
        <canvas id="image-canvas"></canvas>
        <canvas id="overlay-canvas"></canvas>
      </div>

      <div id="loupe">
        <canvas id="loupe-canvas" width="180" height="180"></canvas>
        <div id="loupe-coord">X: 0, Y: 0</div>
      </div>
    </div>

    <div id="sidebar">
      <div class="panel-section">
        <h3>
          <span>Board Alignment</span>
          <div style="display:flex; gap:6px;">
            <button id="btn-wire-grid-calib" style="font-size:0.75rem; padding:3px 8px; background:#22c55e; color:#052e16; font-weight:700;" title="Calibrate 24 exact spider wire intersections">🎯 Wire Grid (24-Pt)</button>
            <button id="btn-apply-calib-early" style="font-size:0.75rem; padding:3px 8px; background:#0284c7; display:none;" title="Apply points clicked so far">✓ Apply</button>
          </div>
        </h3>
        <div style="font-size: 0.8rem; display: flex; flex-direction: column; gap: 4px;">
          <div><strong style="color:var(--text-muted);">ECC Score:</strong> <span id="ecc-score">--</span></div>
          <div><strong style="color:var(--text-muted);">Center (px):</strong> <span id="board-center">--</span></div>
          <div><strong style="color:var(--text-muted);">Fitted Rings:</strong> <span id="board-rings">--</span></div>
          <div id="calib-pts-status" style="color:#38bdf8; font-size:0.75rem; margin-top:2px;"></div>
        </div>
        <div id="adjust-panel" style="margin-top:8px; padding-top:8px; border-top:1px solid var(--card-border); display:flex; flex-direction:column; gap:6px;">
          <div style="display:flex; gap:6px; align-items:center;">
            <button id="btn-refit-wireframe" style="flex:1; padding:4px 8px; font-size:0.75rem; background:#0284c7; color:#fff; font-weight:600;" title="Re-run calibration fit from current points">⚡ Re-Fit</button>
            <button id="btn-save-wireframe" style="flex:1; padding:4px 8px; font-size:0.75rem; background:#10b981; color:#fff; font-weight:700;" title="Save adjusted points to ground truth">💾 Save Grid</button>
            <button id="btn-reset-calib-points" style="padding:4px 8px; font-size:0.75rem; background:#334155; color:#f87171; border:1px solid #475569;" title="Reset points to original">Reset</button>
          </div>
          <div id="adjust-point-info" style="font-size:0.75rem; color:#facc15; min-height:16px;"></div>
          <label style="display:flex; align-items:center; gap:6px; font-size:0.75rem; color:var(--text-muted); cursor:pointer;">
            <input type="checkbox" id="toggle-calib-points" checked style="cursor:pointer;"> Show Point Handles
          </label>
        </div>
      </div>

      <div class="panel-section" style="flex:1;">
        <h3>
          <span>Darts in this Turn</span>
          <span id="total-score-pill" class="badge" style="background:#0369a1; color:#fff;">Total: 0</span>
        </h3>
        <div id="darts-list" style="display:flex; flex-direction:column; gap:8px;"></div>
        <div class="tools-bar" style="margin-top: 8px; display:flex; gap:6px;">
          <button id="btn-add-dart">+ Add Dart</button>
          <button id="btn-undo-click">Undo Click</button>
          <button id="btn-clear-darts" class="btn-danger" style="background:#7f1d1d;">Clear</button>
        </div>
      </div>

      <div class="panel-section">
        <h3>Annotation Mode</h3>
        <p id="mode-hint" style="font-size: 0.8rem; color: var(--accent);">
          Click 1: Dart Tip &bull; Click 2: Dart Tail
        </p>
        <div style="margin-top: 6px; display: flex; align-items: center; gap: 8px;">
          <input type="checkbox" id="chk-verified" style="cursor:pointer;">
          <label for="chk-verified" style="font-size: 0.85rem; font-weight: 600; cursor:pointer;">
            Mark image as Verified
          </label>
        </div>
      </div>

      <div class="panel-section">
        <h3>Shortcuts & Controls</h3>
        <div class="shortcuts">
          <div><kbd>Scroll</kbd> Zoom in/out</div>
          <div><kbd>Right Drag</kbd> / <kbd>Shift+Drag</kbd> Pan around</div>
          <div><kbd>Click Point</kbd> Select handle</div>
          <div><kbd>Arrows</kbd> Nudge point (Shift=5px)</div>
          <div><kbd>R</kbd> Reset pan & zoom</div>
          <div><kbd>O</kbd> Toggle wireframe overlay</div>
          <div><kbd>Z</kbd> Undo point</div>
          <div><kbd>N</kbd> / <kbd>P</kbd> Next / Previous image</div>
          <div><kbd>Space</kbd> / <kbd>Enter</kbd> Save & Next</div>
        </div>
      </div>
    </div>
  </main>

  <!-- MODAL: BENCHMARK & UNSEEN INSPECTOR -->
  <div id="inspector-modal" class="modal-backdrop">
    <div class="modal-content">
      <div class="modal-header">
        <h3 id="modal-title" style="color:#fff; font-size:1.1rem; display:flex; align-items:center; gap:8px;">
          <span>🔍 Visual Comparison Inspector</span>
        </h3>
        <button onclick="closeInspectorModal()" style="padding:4px 10px; background:#475569; color:#fff; border-radius:6px;">✕ Close</button>
      </div>
      <div class="modal-body">
        <div class="modal-image-col">
          <img id="modal-img" src="" alt="Inspection Composite">
        </div>
        <div class="modal-details-col" id="modal-details">
          <!-- Rendered by JS -->
        </div>
      </div>
    </div>
  </div>

  <div id="toast" class="toast">Saved annotations successfully!</div>

  <script>
    // Injected metrics for standalone offline view (replaced when exporting dashboard.html)
    const EMBEDDED_METRICS = /*METRICS_JSON*/ null;

    let state = {
      activeTab: 'dashboard',
      metrics: null,
      unseenData: null,
      currentFilter: 'all',
      dataset: null,
      currentImagePath: null,
      imageObj: new Image(),
      imageLoaded: false,
      calibration: null,
      scale: 1,
      panX: 0,
      panY: 0,
      isPanning: false,
      justPanned: false,
      startX: 0,
      startY: 0,
      pendingTip: null,
      darts: [],
      verified: false,
      showWireframe: true,
      calibMode: false,
      calibClicks: [],
      showCalibPoints: true,
      calibPoints: [],
      initialCalibPoints: [],
      selectedPointIdx: null,
      isDraggingPoint: false,
      draggedPointMoved: false,
      justHandledPoint: false,
      refitTimeout: null
    };

    const viewport = document.getElementById('viewport');
    const container = document.getElementById('canvas-container');
    const imgCanvas = document.getElementById('image-canvas');
    const imgCtx = imgCanvas.getContext('2d');
    const overlayCanvas = document.getElementById('overlay-canvas');
    const overlayCtx = overlayCanvas.getContext('2d');
    const loupe = document.getElementById('loupe');
    const loupeCanvas = document.getElementById('loupe-canvas');
    const loupeCtx = loupeCanvas.getContext('2d');
    const loupeCoord = document.getElementById('loupe-coord');

    const WIRE_GRID_STEPS = [
      { name: "1/24: Spoke 20|1 - Double Outer Wire", spoke: "20|1", ring: "Double Outer" },
      { name: "2/24: Spoke 20|1 - Double Inner Wire", spoke: "20|1", ring: "Double Inner" },
      { name: "3/24: Spoke 20|1 - Treble Outer Wire", spoke: "20|1", ring: "Treble Outer" },
      { name: "4/24: Spoke 20|1 - Treble Inner Wire", spoke: "20|1", ring: "Treble Inner" },
      { name: "5/24: Spoke 20|1 - Outer Bull (Green) Wire", spoke: "20|1", ring: "Outer Bull" },

      { name: "6/24: Spoke 6|10 - Double Outer Wire", spoke: "6|10", ring: "Double Outer" },
      { name: "7/24: Spoke 6|10 - Double Inner Wire", spoke: "6|10", ring: "Double Inner" },
      { name: "8/24: Spoke 6|10 - Treble Outer Wire", spoke: "6|10", ring: "Treble Outer" },
      { name: "9/24: Spoke 6|10 - Treble Inner Wire", spoke: "6|10", ring: "Treble Inner" },
      { name: "10/24: Spoke 6|10 - Outer Bull (Green) Wire", spoke: "6|10", ring: "Outer Bull" },

      { name: "11/24: Spoke 3|19 - Double Outer Wire", spoke: "3|19", ring: "Double Outer" },
      { name: "12/24: Spoke 3|19 - Double Inner Wire", spoke: "3|19", ring: "Double Inner" },
      { name: "13/24: Spoke 3|19 - Treble Outer Wire", spoke: "3|19", ring: "Treble Outer" },
      { name: "14/24: Spoke 3|19 - Treble Inner Wire", spoke: "3|19", ring: "Treble Inner" },
      { name: "15/24: Spoke 3|19 - Outer Bull (Green) Wire", spoke: "3|19", ring: "Outer Bull" },

      { name: "16/24: Spoke 11|14 - Double Outer Wire", spoke: "11|14", ring: "Double Outer" },
      { name: "17/24: Spoke 11|14 - Double Inner Wire", spoke: "11|14", ring: "Double Inner" },
      { name: "18/24: Spoke 11|14 - Treble Outer Wire", spoke: "11|14", ring: "Treble Outer" },
      { name: "19/24: Spoke 11|14 - Treble Inner Wire", spoke: "11|14", ring: "Treble Inner" },
      { name: "20/24: Spoke 11|14 - Outer Bull (Green) Wire", spoke: "11|14", ring: "Outer Bull" },

      { name: "21/24: Inner Bull Wire on Spoke 20|1", spoke: "20|1", ring: "Inner Bull" },
      { name: "22/24: Inner Bull Wire on Spoke 6|10", spoke: "6|10", ring: "Inner Bull" },
      { name: "23/24: Inner Bull Wire on Spoke 3|19", spoke: "3|19", ring: "Inner Bull" },
      { name: "24/24: Inner Bull Wire on Spoke 11|14", spoke: "11|14", ring: "Inner Bull" }
    ];

    async function init() {
      setupTabSwitching();
      setupEvents();

      // Load initial dashboard metrics
      if (EMBEDDED_METRICS) {
        state.metrics = EMBEDDED_METRICS;
        renderDashboard();
      } else {
        await loadDashboardMetrics();
      }

      await loadUnseenGallery();

      // Preload Annotator dataset in background
      try {
        const res = await fetch('/api/dataset');
        state.dataset = await res.json();
        populateSelect();
      } catch (e) {
        console.warn('Dataset load deferred', e);
      }
    }

    /* Tab Switching */
    function setupTabSwitching() {
      const btnDash = document.getElementById('tab-btn-dashboard');
      const btnAnn = document.getElementById('tab-btn-annotator');
      const viewDash = document.getElementById('dashboard-view');
      const viewAnn = document.getElementById('annotator-view');
      const ctrlDash = document.getElementById('dash-header-controls');
      const ctrlAnn = document.getElementById('annotator-header-controls');

      btnDash.addEventListener('click', () => {
        state.activeTab = 'dashboard';
        btnDash.className = 'nav-tab active';
        btnAnn.className = 'nav-tab';
        viewDash.style.display = 'flex';
        viewAnn.style.display = 'none';
        ctrlDash.style.display = 'flex';
        ctrlAnn.style.display = 'none';
      });

      btnAnn.addEventListener('click', () => {
        state.activeTab = 'annotator';
        btnAnn.className = 'nav-tab active';
        btnDash.className = 'nav-tab';
        viewDash.style.display = 'none';
        viewAnn.style.display = 'flex';
        ctrlDash.style.display = 'none';
        ctrlAnn.style.display = 'flex';

        if (!state.currentImagePath && state.dataset && state.dataset.starter_15.length > 0) {
          loadImage(state.dataset.starter_15[0]);
        } else if (state.imageLoaded) {
          resetTransform();
        }
      });

      document.getElementById('btn-rerun-eval').addEventListener('click', triggerRerunEval);
    }

    /* Dashboard Metrics Loader & Renderer */
    async function loadDashboardMetrics() {
      try {
        const res = await fetch('/api/eval_metrics');
        state.metrics = await res.json();
        renderDashboard();
      } catch (err) {
        console.error('Failed to load eval metrics', err);
      }
    }

    async function triggerRerunEval() {
      const btn = document.getElementById('btn-rerun-eval');
      const origText = btn.innerHTML;
      btn.innerHTML = '<span>⏳ Evaluating...</span>';
      btn.disabled = true;

      try {
        const res = await fetch('/api/run_eval', { method: 'POST' });
        state.metrics = await res.json();
        renderDashboard();
        showToast('Evaluation benchmark complete!');
      } catch (err) {
        alert('Evaluation run failed: ' + err);
      } finally {
        btn.innerHTML = origText;
        btn.disabled = false;
      }
    }

    function renderDashboard() {
      const m = state.metrics;
      if (!m) return;

      // Section A: KPI Cards
      const dm = m.detection_metrics || {};
      document.getElementById('kpi-f1').textContent = `${dm.f1_score_pct || 0}%`;
      document.getElementById('kpi-prec').textContent = `${dm.precision_pct || 0}%`;
      document.getElementById('kpi-rec').textContent = `${dm.recall_pct || 0}%`;
      document.getElementById('kpi-count-acc').textContent = `${dm.count_accuracy_pct || 0}%`;
      document.getElementById('kpi-f1-bar').style.width = `${Math.min(100, dm.f1_score_pct || 0)}%`;

      const sc = m.scoring_accuracy || {};
      document.getElementById('kpi-label-acc').textContent = `${sc.label_accuracy_pct || 0}%`;
      document.getElementById('kpi-sub-label').textContent = `${sc.label_accuracy_pct || 0}%`;
      document.getElementById('kpi-pts-acc').textContent = `${sc.point_accuracy_pct || 0}%`;
      document.getElementById('kpi-turn-acc').textContent = `${sc.turn_total_score_accuracy_pct || 0}%`;
      document.getElementById('kpi-score-bar').style.width = `${Math.min(100, sc.label_accuracy_pct || 0)}%`;

      const tip = m.tip_localization_error || {};
      document.getElementById('kpi-tip-med').textContent = `${tip.median_mm || 0} mm`;
      document.getElementById('kpi-tip-mean').textContent = `${tip.mean_mm || 0} mm`;
      document.getElementById('kpi-tip-p90').textContent = `${tip.p90_mm || 0} mm`;
      document.getElementById('kpi-tip-px').textContent = `${tip.median_px || 0} px`;

      const bp = m.board_pose_accuracy || {};
      document.getElementById('kpi-wire-err').textContent = `${bp.mean_24pt_wire_error_mm || 0} mm`;
      document.getElementById('kpi-wire-px').textContent = `${bp.mean_24pt_wire_error_px || 0} px`;
      document.getElementById('kpi-corner-err').textContent = `${bp.mean_corner_wire_error_mm || 0} mm`;
      document.getElementById('kpi-center-err').textContent = `${bp.mean_center_error_px || 0} px (${bp.mean_center_error_mm || 0} mm)`;

      const lat = m.runtime_latency_ms || {};
      document.getElementById('kpi-total-lat').textContent = `${Math.round(lat.total_mean || 0)} ms`;
      document.getElementById('kpi-board-ms').textContent = `${Math.round(lat.board_mean || 0)} ms`;
      document.getElementById('kpi-darts-ms').textContent = `${Math.round(lat.darts_mean || 0)} ms`;
      document.getElementById('kpi-p95-lat').textContent = `${Math.round(lat.total_p95 || 0)} ms`;

      const tot = (lat.board_mean || 1) + (lat.darts_mean || 1);
      document.getElementById('kpi-board-lat-bar').style.width = `${Math.round((lat.board_mean || 0) / tot * 100)}%`;
      document.getElementById('kpi-darts-lat-bar').style.width = `${Math.round((lat.darts_mean || 0) / tot * 100)}%`;

      // Section B: 15-Image Benchmark Table
      renderBenchmarkTable();
    }

    function filterBenchmark(cat) {
      state.currentFilter = cat;
      const pills = document.querySelectorAll('.filter-pill');
      pills.forEach(p => {
        if (p.textContent.toLowerCase().includes(cat)) p.className = 'filter-pill active';
        else if (cat === 'all' && p.textContent.includes('All')) p.className = 'filter-pill active';
        else p.className = 'filter-pill';
      });
      renderBenchmarkTable();
    }

    function renderBenchmarkTable() {
      const tbody = document.getElementById('bench-table-body');
      if (!tbody || !state.metrics || !state.metrics.per_image) return;

      const items = state.metrics.per_image.filter(item => {
        if (state.currentFilter === 'all') return true;
        if (state.currentFilter === 'pass') return item.status === 'PASS';
        if (state.currentFilter === 'fail') return item.status === 'FAIL';
        return item.category === state.currentFilter;
      });

      tbody.innerHTML = '';
      items.forEach((item, idx) => {
        const tr = document.createElement('tr');
        const passBadge = item.status === 'PASS' 
          ? '<span class="badge badge-pass">PASS</span>' 
          : '<span class="badge badge-fail">FAIL</span>';

        const catBadge = item.category === 'raw_board'
          ? '<span class="badge" style="background:#334155; color:#38bdf8;">Raw Board</span>'
          : (item.category === 'single_dart' ? '<span class="badge" style="background:#1e293b; color:#22c55e;">Single</span>' : '<span class="badge" style="background:#1e293b; color:#a855f7;">Multi</span>');

        const gtLabelsStr = item.gt_labels && item.gt_labels.length > 0 ? item.gt_labels.join(', ') : 'None';
        const predLabelsStr = item.pred_labels && item.pred_labels.length > 0 ? item.pred_labels.join(', ') : 'None';

        const wgError = item.board_alignment ? `${item.board_alignment.wire_grid_24pt_error_mm} mm` : '--';

        tr.innerHTML = `
          <td style="color:var(--text-muted);">${idx + 1}</td>
          <td><strong>${item.image}</strong></td>
          <td>${catBadge}</td>
          <td>${item.gt_darts} / ${item.pred_darts}</td>
          <td>GT: ${item.gt_score} (${gtLabelsStr}) &bull; Pred: ${item.pred_score} (${predLabelsStr})</td>
          <td>${item.mean_error_mm > 0 ? item.mean_error_mm.toFixed(1) + ' mm' : '--'}</td>
          <td style="color:#a855f7;">${wgError}</td>
          <td>${Math.round(item.total_ms || 0)} ms</td>
          <td>${passBadge}</td>
          <td><button class="btn-primary" style="font-size:0.75rem; padding:3px 8px;" onclick="openBenchmarkModal(${idx})">🔍 Inspect</button></td>
        `;
        tbody.appendChild(tr);
      });
    }

    window.openBenchmarkModal = function(idx) {
      if (!state.metrics || !state.metrics.per_image) return;
      const item = state.metrics.per_image[idx];
      if (!item) return;

      document.getElementById('modal-title').innerHTML = `<span>🔍 Inspection: ${item.image}</span> <span class="badge ${item.status === 'PASS' ? 'badge-pass' : 'badge-fail'}">${item.status}</span>`;
      document.getElementById('modal-img').src = `/api/composite?file=${encodeURIComponent(item.composite_file)}`;

      const details = document.getElementById('modal-details');
      let dartRows = '';
      if (item.gt_darts_detail && item.gt_darts_detail.length > 0) {
        item.gt_darts_detail.forEach((gd, di) => {
          dartRows += `<tr><td>GT #${di+1}</td><td>${gd.label} (${gd.score} pts)</td><td>Tip: ${Math.round(gd.tip_px[0])}, ${Math.round(gd.tip_px[1])}</td></tr>`;
        });
      }
      if (item.pred_darts_detail && item.pred_darts_detail.length > 0) {
        item.pred_darts_detail.forEach((pd, di) => {
          dartRows += `<tr><td style="color:#4ade80;">Pred #${di+1}</td><td style="color:#4ade80;">${pd.label} (${pd.score} pts, conf:${pd.confidence})</td><td>Tip: ${Math.round(pd.tip_px[0])}, ${Math.round(pd.tip_px[1])}</td></tr>`;
        });
      }

      let boardHtml = '';
      if (item.board_alignment) {
        const ba = item.board_alignment;
        boardHtml = `
          <div class="panel-section">
            <h4 style="color:#a855f7; font-size:0.85rem;">24-Point Wire Grid Alignment</h4>
            <div style="font-size:0.8rem; display:flex; flex-direction:column; gap:4px; margin-top:4px;">
              <div>24-Pt Mean Error: <strong>${ba.wire_grid_24pt_error_mm} mm (${ba.wire_grid_24pt_error_px} px)</strong></div>
              <div>Corner Wire Error: <strong>${ba.corner_wire_error_mm} mm (${ba.corner_wire_error_px} px)</strong></div>
              <div>Bull Center Error: <strong>${ba.center_error_mm} mm (${ba.center_error_px} px)</strong></div>
              <div>ECC Fit Quality: <strong>${ba.ecc_score}</strong></div>
            </div>
          </div>
        `;
      }

      details.innerHTML = `
        <div class="panel-section">
          <h4>Benchmark Summary</h4>
          <div style="font-size:0.85rem; display:flex; flex-direction:column; gap:4px; margin-top:4px;">
            <div>Category: <strong>${item.category}</strong></div>
            <div>Ground Truth Score: <strong>${item.gt_score}</strong> (${item.gt_darts} darts)</div>
            <div>Pipeline Predicted Score: <strong>${item.pred_score}</strong> (${item.pred_darts} darts)</div>
            <div>Tip Euclidean Error: <strong>${item.mean_error_mm.toFixed(2)} mm (${item.mean_error_px.toFixed(1)} px)</strong></div>
            <div>Stage Latencies: <strong>Board ${item.board_ms} ms &bull; Darts ${item.darts_ms} ms &bull; Total ${item.total_ms} ms</strong></div>
          </div>
        </div>
        ${boardHtml}
        <div class="panel-section" style="flex:1;">
          <h4>Dart Match Details</h4>
          <table style="width:100%; font-size:0.8rem; border-collapse:collapse; margin-top:6px;">
            ${dartRows || '<tr><td style="color:var(--text-muted);">No darts in this turn (raw board)</td></tr>'}
          </table>
        </div>
      `;

      document.getElementById('inspector-modal').style.display = 'flex';
    };

    window.closeInspectorModal = function() {
      document.getElementById('inspector-modal').style.display = 'none';
    };

    /* SECTION C: Unseen Images Gallery */
    async function loadUnseenGallery() {
      try {
        const res = await fetch('/api/unannotated_images');
        state.unseenData = await res.json();
        const sel = document.getElementById('unseen-select');
        sel.innerHTML = '<option value="">-- Choose from 63 Unseen Images --</option>';

        ['raw_board', 'single_dart', 'multi_dart'].forEach(cat => {
          const optg = document.createElement('optgroup');
          optg.label = cat.replace('_', ' ').toUpperCase() + ` (${state.unseenData.categories[cat].length})`;
          state.unseenData.categories[cat].forEach(p => {
            const opt = document.createElement('option');
            opt.value = p;
            const parts = p.split('/');
            opt.textContent = parts[parts.length - 1];
            optg.appendChild(opt);
          });
          sel.appendChild(optg);
        });

        renderUnseenCards();
      } catch (err) {
        console.warn('Failed to load unseen images', err);
      }
    }

    function renderUnseenCards() {
      const gal = document.getElementById('unseen-gallery');
      if (!gal || !state.unseenData || !state.unseenData.starter_9) return;

      gal.innerHTML = '';
      state.unseenData.starter_9.forEach((imgPath, idx) => {
        const parts = imgPath.split('/');
        const fname = parts[parts.length - 1];
        const cat = parts[parts.length - 2];

        const card = document.createElement('div');
        card.className = 'gallery-card';
        card.id = `unseen-card-${idx}`;
        card.innerHTML = `
          <div class="gallery-thumb">
            <img src="/api/image?path=${encodeURIComponent(imgPath)}" alt="${fname}" loading="lazy">
          </div>
          <div class="gallery-info">
            <div class="gallery-title">
              <span>${fname}</span>
              <span class="badge" style="background:#334155; color:#38bdf8;">${cat}</span>
            </div>
            <div id="unseen-res-${idx}" class="gallery-results">
              <span>Ready for live inference</span>
            </div>
            <div class="gallery-actions">
              <button class="btn-primary" style="flex:1; padding:5px 8px; font-size:0.75rem;" onclick="runUnseenLive('${imgPath}', ${idx})">⚡ Run CV Pipeline</button>
              <button class="btn-secondary" style="padding:5px 8px; font-size:0.75rem;" onclick="openUnseenModal('${imgPath}')">🔍 Overlay</button>
            </div>
          </div>
        `;
        gal.appendChild(card);
      });
    }

    window.runUnseenLive = async function(path, cardIdx) {
      const resEl = document.getElementById(`unseen-res-${cardIdx}`);
      if (resEl) resEl.innerHTML = '<span>⏳ Running detector...</span>';

      try {
        const res = await fetch(`/api/analyze_live?path=${encodeURIComponent(path)}`);
        const data = await res.json();

        let dartsSummary = `${data.darts.length} dart(s)`;
        if (data.darts.length > 0) {
          dartsSummary += `: ` + data.darts.map(d => `${d.label} (${d.score})`).join(', ');
        }

        if (resEl) {
          resEl.innerHTML = `
            <div><strong>Score: ${data.total_score}</strong> (${dartsSummary})</div>
            <div style="font-size:0.75rem; color:var(--text-muted);">Latency: ${data.timings_ms.total_ms} ms (Board: ${data.timings_ms.board_ms} ms, Darts: ${data.timings_ms.darts_ms} ms)</div>
          `;
        }
      } catch (err) {
        if (resEl) resEl.innerHTML = `<span style="color:#f87171;">Failed: ${err}</span>`;
      }
    };

    window.analyzeSelectedUnseen = function(path) {
      if (!path) return;
      openUnseenModal(path);
    };

    window.openUnseenModal = function(path) {
      const parts = path.split('/');
      const fname = parts[parts.length - 1];
      document.getElementById('modal-title').innerHTML = `<span>🔭 Unseen Live Analysis: ${fname}</span>`;
      document.getElementById('modal-img').src = `/api/render_analysis?path=${encodeURIComponent(path)}&t=${Date.now()}`;

      const details = document.getElementById('modal-details');
      details.innerHTML = `<div style="color:var(--text-muted); padding:20px;">Computing live board homography and dart detections...</div>`;

      document.getElementById('inspector-modal').style.display = 'flex';

      fetch(`/api/analyze_live?path=${encodeURIComponent(path)}`)
        .then(r => r.json())
        .then(data => {
          let dartRows = '';
          data.darts.forEach((d, di) => {
            dartRows += `<tr><td>Dart #${di+1}</td><td><strong>${d.label}</strong> (${d.score} pts)</td><td>Conf: ${d.confidence}</td><td>Tip: ${d.tip_px[0]}, ${d.tip_px[1]}</td></tr>`;
          });

          details.innerHTML = `
            <div class="panel-section">
              <h4>Detection Result</h4>
              <div style="font-size:0.85rem; display:flex; flex-direction:column; gap:4px; margin-top:4px;">
                <div>Predicted Total Score: <strong style="color:#38bdf8; font-size:1.1rem;">${data.total_score} pts</strong></div>
                <div>Darts Detected: <strong>${data.darts.length}</strong></div>
                <div>ECC Pose Score: <strong>${data.board ? data.board.ecc_score : 'N/A'}</strong></div>
                <div>Board Center: <strong>${data.board ? data.board.center[0] + ', ' + data.board.center[1] : 'N/A'}</strong></div>
                <div>Stage Latencies: <strong>Board ${data.timings_ms.board_ms} ms &bull; Darts ${data.timings_ms.darts_ms} ms &bull; Total ${data.timings_ms.total_ms} ms</strong></div>
              </div>
            </div>
            <div class="panel-section" style="flex:1;">
              <h4>Detected Darts Telemetry</h4>
              <table style="width:100%; font-size:0.8rem; border-collapse:collapse; margin-top:6px;">
                ${dartRows || '<tr><td style="color:var(--text-muted);">No darts detected (clean board)</td></tr>'}
              </table>
            </div>
          `;
        })
        .catch(err => {
          details.innerHTML = `<div style="color:#f87171;">Error analyzing image: ${err}</div>`;
        });
    };

    /* ANNOTATOR JAVASCRIPT LOGIC */
    function populateSelect() {
      const sel = document.getElementById('image-select');
      sel.innerHTML = '';
      state.dataset.starter_15.forEach((path, idx) => {
        const opt = document.createElement('option');
        opt.value = path;
        const entry = state.dataset.images[path];
        const vTag = entry && entry.verified ? '✓ ' : '⏳ ';
        const dCount = entry && entry.darts ? entry.darts.length : 0;
        const parts = path.split('/');
        opt.textContent = `${vTag}${idx+1}. ${parts[parts.length-2]}/${parts[parts.length-1]} (${dCount} darts)`;
        sel.appendChild(opt);
      });
      sel.addEventListener('change', (e) => loadImage(e.target.value));
    }

    async function loadImage(path) {
      state.currentImagePath = path;
      document.getElementById('image-select').value = path;
      const idx = state.dataset.starter_15.indexOf(path);
      document.getElementById('image-progress').textContent = `${idx + 1} / ${state.dataset.starter_15.length}`;

      const entry = state.dataset.images[path] || { darts: [], verified: false, total_score: 0 };
      state.darts = JSON.parse(JSON.stringify(entry.darts || []));
      state.verified = !!entry.verified;
      state.pendingTip = null;
      state.calibMode = false;
      state.calibClicks = [];
      document.getElementById('btn-apply-calib-early').style.display = 'none';
      document.getElementById('chk-verified').checked = state.verified;
      updateVerifiedBadge();

      // Check if calibration points exist
      const calStatus = document.getElementById('calib-pts-status');
      if (entry.wire_grid_points_px && entry.wire_grid_points_px.length >= 4) {
        calStatus.textContent = `✓ Wire-grid (${entry.wire_grid_points_px.length} pts) calibrated`;
        state.calibPoints = JSON.parse(JSON.stringify(entry.wire_grid_points_px));
      } else if (entry.calibration_points_px && entry.calibration_points_px.length === 4) {
        calStatus.textContent = "✓ 4-point wireframe calibrated";
        state.calibPoints = JSON.parse(JSON.stringify(entry.calibration_points_px));
      } else {
        calStatus.textContent = "";
        state.calibPoints = [];
      }
      state.initialCalibPoints = JSON.parse(JSON.stringify(state.calibPoints));
      state.selectedPointIdx = null;

      // Fetch calibration
      const calRes = await fetch(`/api/calibration?path=${encodeURIComponent(path)}`);
      state.calibration = await calRes.json();
      document.getElementById('ecc-score').textContent = state.calibration.ecc_score ? state.calibration.ecc_score.toFixed(4) : 'N/A';
      document.getElementById('board-center').textContent = state.calibration.center ? `${Math.round(state.calibration.center[0])}, ${Math.round(state.calibration.center[1])}` : 'N/A';
      document.getElementById('board-rings').textContent = state.calibration.rings_count ? `${state.calibration.rings_count}/4 paint edges fitted` : 'Exact wire-grid rings';

      if ((!state.calibPoints || state.calibPoints.length === 0) && state.calibration) {
        if (state.calibration.wire_grid_points) {
          state.calibPoints = JSON.parse(JSON.stringify(state.calibration.wire_grid_points));
        } else if (state.calibration.calib_points) {
          state.calibPoints = JSON.parse(JSON.stringify(state.calibration.calib_points));
        }
        state.initialCalibPoints = JSON.parse(JSON.stringify(state.calibPoints));
      }
      updatePointInfo();

      // Load image bitmap
      state.imageLoaded = false;
      state.imageObj = new Image();
      state.imageObj.onload = () => {
        imgCanvas.width = state.imageObj.width;
        imgCanvas.height = state.imageObj.height;
        overlayCanvas.width = state.imageObj.width;
        overlayCanvas.height = state.imageObj.height;
        imgCtx.drawImage(state.imageObj, 0, 0);
        state.imageLoaded = true;
        resetTransform();
        renderOverlay();
        renderDartsList();
      };
      state.imageObj.src = `/api/image?path=${encodeURIComponent(path)}`;
    }

    function resetTransform() {
      if (!state.imageLoaded) return;
      const vw = viewport.clientWidth;
      const vh = viewport.clientHeight;
      const iw = imgCanvas.width;
      const ih = imgCanvas.height;
      state.scale = Math.min(vw / iw, vh / ih) * 0.95;
      state.panX = (vw - iw * state.scale) / 2;
      state.panY = (vh - ih * state.scale) / 2;
      applyTransform();
    }

    function applyTransform() {
      container.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.scale})`;
    }

    function updateVerifiedBadge() {
      const b = document.getElementById('verified-badge');
      if (state.verified) {
        b.className = 'badge badge-verified';
        b.textContent = 'Verified';
      } else {
        b.className = 'badge badge-unverified';
        b.textContent = 'Unverified';
      }
    }

    function renderOverlay() {
      if (!state.imageLoaded) return;
      const ctx = overlayCtx;
      ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

      // Draw wireframe if enabled
      if (state.showWireframe && state.calibration && state.calibration.wires) {
        ctx.save();
        ctx.lineWidth = Math.max(2, Math.round(imgCanvas.width / 1600));
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.45)';
        for (const wire of state.calibration.wires) {
          ctx.beginPath();
          wire.points.forEach((pt, i) => {
            if (i === 0) ctx.moveTo(pt[0], pt[1]);
            else ctx.lineTo(pt[0], pt[1]);
          });
          if (wire.type === 'circle') ctx.closePath();
          ctx.stroke();
        }
        ctx.restore();
      }

      // Draw Calibration Points Handles
      if (!state.calibMode && state.showCalibPoints && state.calibPoints && state.calibPoints.length > 0) {
        const scale = Math.max(1, Math.round(imgCanvas.width / 2200));
        ctx.save();
        state.calibPoints.forEach((pt, i) => {
          const isSelected = (state.selectedPointIdx === i);
          let pointColor = '#10b981';
          if (isSelected) {
            pointColor = '#f59e0b';
          } else if (i === 0 || i === 5 || i === 10 || i === 15) {
            pointColor = '#3b82f6';
          } else if (i % 5 === 1) {
            pointColor = '#10b981';
          } else if (i % 5 === 2 || i % 5 === 3) {
            pointColor = '#06b6d4';
          } else if (i % 5 === 4) {
            pointColor = '#84cc16';
          } else if (i >= 20) {
            pointColor = '#ef4444';
          }

          ctx.beginPath();
          ctx.arc(pt[0], pt[1], (isSelected ? 11 : 7) * scale, 0, Math.PI * 2);
          ctx.fillStyle = pointColor;
          ctx.fill();
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = (isSelected ? 3 : 2) * scale;
          ctx.stroke();

          ctx.beginPath();
          ctx.arc(pt[0], pt[1], 1.5 * scale, 0, Math.PI * 2);
          ctx.fillStyle = '#ffffff';
          ctx.fill();

          ctx.font = `bold ${Math.max(12, 11 * scale)}px sans-serif`;
          ctx.fillStyle = isSelected ? '#fbbf24' : '#ffffff';
          ctx.shadowColor = '#000000';
          ctx.shadowBlur = 4;
          ctx.fillText(`${i + 1}`, pt[0] + 9 * scale, pt[1] - 5 * scale);
          ctx.shadowBlur = 0;
        });
        ctx.restore();
      }

      // Draw darts
      state.darts.forEach((d, idx) => {
        const tip = d.tip_px;
        const tail = d.tail_px;
        const scale = Math.max(1, Math.round(imgCanvas.width / 2000));

        if (tail) {
          ctx.beginPath();
          ctx.strokeStyle = '#eab308';
          ctx.lineWidth = 4 * scale;
          ctx.moveTo(tail[0], tail[1]);
          ctx.lineTo(tip[0], tip[1]);
          ctx.stroke();

          ctx.beginPath();
          ctx.fillStyle = '#facc15';
          ctx.arc(tail[0], tail[1], 7 * scale, 0, Math.PI * 2);
          ctx.fill();
        }

        ctx.beginPath();
        ctx.arc(tip[0], tip[1], 10 * scale, 0, Math.PI * 2);
        ctx.strokeStyle = '#22c55e';
        ctx.lineWidth = 3 * scale;
        ctx.stroke();

        ctx.beginPath();
        ctx.fillStyle = '#4ade80';
        ctx.arc(tip[0], tip[1], 4 * scale, 0, Math.PI * 2);
        ctx.fill();

        ctx.font = `bold ${16 * scale}px sans-serif`;
        const text = `Dart ${idx + 1}: ${d.label || 'Dart'} (${d.score || 0})`;
        const tw = ctx.measureText(text).width;
        ctx.fillStyle = 'rgba(0,0,0,0.75)';
        ctx.fillRect(tip[0] + 14 * scale, tip[1] - 22 * scale, tw + 12 * scale, 26 * scale);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(text, tip[0] + 20 * scale, tip[1] - 3 * scale);
      });

      if (state.pendingTip) {
        const tip = state.pendingTip;
        const scale = Math.max(1, Math.round(imgCanvas.width / 2000));
        ctx.beginPath();
        ctx.arc(tip[0], tip[1], 12 * scale, 0, Math.PI * 2);
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 4 * scale;
        ctx.stroke();
        ctx.beginPath();
        ctx.fillStyle = '#0284c7';
        ctx.arc(tip[0], tip[1], 5 * scale, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    function parseDartLabel(raw) {
      if (!raw) return null;
      let s = raw.toUpperCase().trim();
      if (s === 'DBULL' || s === 'D50') return { label: 'DBULL', sector: 50, multiplier: 2, score: 50 };
      if (s === 'BULL' || s === 'S25') return { label: 'BULL', sector: 25, multiplier: 1, score: 25 };
      if (s === 'MISS' || s === '0') return { label: 'MISS', sector: 0, multiplier: 0, score: 0 };
      const m = s.match(/^([SDT]?)(20|19|18|17|16|15|14|13|12|11|10|[1-9])$/);
      if (m) {
        const type = m[1] || 'S';
        const sec = parseInt(m[2], 10);
        const mult = type === 'T' ? 3 : (type === 'D' ? 2 : 1);
        return { label: `${type}${sec}`, sector: sec, multiplier: mult, score: sec * mult };
      }
      return null;
    }

    window.updateDartLabel = function(idx, val) {
      const parsed = parseDartLabel(val);
      if (parsed) {
        state.darts[idx].label = parsed.label;
        state.darts[idx].sector = parsed.sector;
        state.darts[idx].multiplier = parsed.multiplier;
        state.darts[idx].score = parsed.score;
      } else {
        state.darts[idx].label = val.toUpperCase().trim();
      }
      renderOverlay();
      renderDartsList();
    };

    window.updateDartScore = function(idx, val) {
      state.darts[idx].score = parseInt(val, 10) || 0;
      renderOverlay();
      renderDartsList();
    };

    function renderDartsList() {
      const list = document.getElementById('darts-list');
      list.innerHTML = '';
      let total = 0;

      state.darts.forEach((d, idx) => {
        total += (d.score || 0);
        const card = document.createElement('div');
        card.className = 'dart-card';
        card.innerHTML = `
          <div class="dart-header">
            <span>Dart #${idx + 1}</span>
            <div style="display:flex; align-items:center; gap:6px;">
              <input type="text" style="width:65px; font-weight:700;" value="${d.label || 'S0'}" onchange="updateDartLabel(${idx}, this.value)">
              <input type="number" style="width:55px;" value="${d.score || 0}" onchange="updateDartScore(${idx}, this.value)">
              <span style="font-size:0.75rem; color:var(--text-muted);">pts</span>
            </div>
          </div>
          <div class="coord-grid">
            <div class="coord-item">
              <label>Tip (X, Y)</label>
              <input type="text" value="${Math.round(d.tip_px[0])}, ${Math.round(d.tip_px[1])}" readonly>
            </div>
            <div class="coord-item">
              <label>Tail (X, Y)</label>
              <input type="text" value="${d.tail_px ? Math.round(d.tail_px[0]) + ', ' + Math.round(d.tail_px[1]) : 'None'}" readonly>
            </div>
          </div>
          <div style="display:flex; justify-content:flex-end; gap:6px; margin-top:4px;">
            <button class="btn-danger" style="font-size:0.75rem; padding:3px 8px;" onclick="deleteDart(${idx})">Delete</button>
          </div>
        `;
        list.appendChild(card);
      });

      document.getElementById('total-score-pill').textContent = `Total: ${total}`;
    }

    window.deleteDart = function(idx) {
      state.darts.splice(idx, 1);
      renderOverlay();
      renderDartsList();
    };

    function getHitPointIndex(imgPt) {
      if (!state.showCalibPoints || !state.calibPoints || state.calibPoints.length === 0) return -1;
      const screenRadius = 22;
      const hitRadius = Math.max(8, screenRadius / state.scale);
      let bestIdx = -1;
      let bestDist = hitRadius;
      for (let i = 0; i < state.calibPoints.length; i++) {
        const p = state.calibPoints[i];
        const d = Math.hypot(imgPt.x - p[0], imgPt.y - p[1]);
        if (d <= bestDist) {
          bestDist = d;
          bestIdx = i;
        }
      }
      return bestIdx;
    }

    function updatePointInfo() {
      const infoEl = document.getElementById('adjust-point-info');
      if (!infoEl) return;
      if (state.selectedPointIdx !== null && state.calibPoints && state.calibPoints[state.selectedPointIdx]) {
        const idx = state.selectedPointIdx;
        const pt = state.calibPoints[idx];
        const stepName = WIRE_GRID_STEPS[idx] ? WIRE_GRID_STEPS[idx].name : `Point ${idx + 1}`;
        infoEl.innerHTML = `<strong style="color:#fbbf24;">#${idx + 1}: ${stepName}</strong><br><span style="color:#94a3b8;">(${pt[0]}, ${pt[1]}) &bull; Drag or Arrow keys to nudge</span>`;
      } else if (state.calibPoints && state.calibPoints.length > 0) {
        infoEl.innerHTML = `<span style="color:#94a3b8;">Click any of ${state.calibPoints.length} points to adjust</span>`;
      } else {
        infoEl.innerHTML = '';
      }
    }

    async function recalibrateWithPoints(points, saveToDisk = false) {
      if (!points || points.length < 4) return;
      document.getElementById('mode-hint').innerHTML = '<span style="color:#22c55e;">Recalibrating wireframe (Joint Projective ALS)...</span>';
      try {
        const calRes = await fetch('/api/recalibrate_wire_grid', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            path: state.currentImagePath,
            points: points,
            save: saveToDisk
          })
        });
        state.calibration = await calRes.json();
        document.getElementById('ecc-score').textContent = state.calibration.ecc_score ? state.calibration.ecc_score.toFixed(4) : 'N/A';
        document.getElementById('board-center').textContent = state.calibration.center ? `${Math.round(state.calibration.center[0])}, ${Math.round(state.calibration.center[1])}` : 'N/A';
        document.getElementById('calib-pts-status').textContent = `✓ Wire-grid (${points.length} pts) calibrated`;
        document.getElementById('mode-hint').innerHTML = 'Click 1: Dart Tip &bull; Click 2: Dart Tail';
        renderOverlay();
        if (saveToDisk) {
          state.initialCalibPoints = JSON.parse(JSON.stringify(points));
          showToast(`Saved ${points.length}-point wireframe to ground truth!`);
        }
      } catch (err) {
        console.error('Recalibration error:', err);
      }
    }

    async function applyWireGridCalibration() {
      if (state.calibClicks.length < 4) {
        alert("Need at least 4 points to compute homography!");
        return;
      }
      document.getElementById('mode-hint').innerHTML = '<span style="color:#22c55e;">Fitting Joint Projective ALS model...</span>';
      const calRes = await fetch('/api/recalibrate_wire_grid', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          path: state.currentImagePath,
          points: state.calibClicks,
          save: true
        })
      });
      state.calibration = await calRes.json();
      state.calibPoints = JSON.parse(JSON.stringify(state.calibClicks));
      state.initialCalibPoints = JSON.parse(JSON.stringify(state.calibPoints));
      state.calibMode = false;
      document.getElementById('btn-apply-calib-early').style.display = 'none';
      document.getElementById('calib-pts-status').textContent = `✓ Wire-grid (${state.calibClicks.length} pts) calibrated`;
      document.getElementById('mode-hint').innerHTML = 'Click 1: Dart Tip &bull; Click 2: Dart Tail';
      updatePointInfo();
      renderOverlay();
      showToast(`Exact wire-grid alignment applied (${state.calibClicks.length} pts)!`);
    }

    function setupEvents() {
      // Pan & Zoom
      viewport.addEventListener('wheel', (e) => {
        e.preventDefault();
        const rect = viewport.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const zoomFactor = e.deltaY < 0 ? 1.18 : 1 / 1.18;
        const newScale = Math.max(0.05, Math.min(25, state.scale * zoomFactor));

        state.panX = mouseX - (mouseX - state.panX) * (newScale / state.scale);
        state.panY = mouseY - (mouseY - state.panY) * (newScale / state.scale);
        state.scale = newScale;
        applyTransform();
        updateLoupe(e);
      }, { passive: false });

      // Disable context menu on viewport to allow holding right mouse to pan
      viewport.addEventListener('contextmenu', (e) => {
        e.preventDefault();
      });

      viewport.addEventListener('mousedown', (e) => {
        if (e.button === 2 || e.button === 1 || e.shiftKey) { // right click, middle click, or Shift+drag
          e.preventDefault();
          state.isPanning = true;
          state.startX = e.clientX - state.panX;
          state.startY = e.clientY - state.panY;
          viewport.style.cursor = 'grabbing';
          return;
        }

        if (e.button === 0 && !state.calibMode && state.showCalibPoints && state.imageLoaded) {
          const pt = getImagePoint(e);
          const hitIdx = getHitPointIndex(pt);
          if (hitIdx !== -1) {
            state.selectedPointIdx = hitIdx;
            state.isDraggingPoint = true;
            state.draggedPointMoved = false;
            state.justHandledPoint = true;
            viewport.style.cursor = 'grabbing';
            updatePointInfo();
            renderOverlay();
            return;
          }
        }
      });

      window.addEventListener('mousemove', (e) => {
        if (state.isPanning) {
          state.panX = e.clientX - state.startX;
          state.panY = e.clientY - state.startY;
          applyTransform();
        } else if (state.isDraggingPoint && state.selectedPointIdx !== null) {
          const pt = getImagePoint(e);
          state.calibPoints[state.selectedPointIdx] = [Math.round(pt.x), Math.round(pt.y)];
          state.draggedPointMoved = true;
          updatePointInfo();
          renderOverlay();
        } else if (!state.calibMode && state.showCalibPoints && state.imageLoaded) {
          const pt = getImagePoint(e);
          const hitIdx = getHitPointIndex(pt);
          if (hitIdx !== -1) {
            viewport.style.cursor = 'grab';
          } else if (viewport.style.cursor === 'grab') {
            viewport.style.cursor = 'crosshair';
          }
        }
        updateLoupe(e);
      });

      window.addEventListener('mouseup', (e) => {
        if (state.isPanning) {
          state.isPanning = false;
          state.justPanned = true;
          setTimeout(() => { state.justPanned = false; }, 80);
          viewport.style.cursor = 'crosshair';
        }
        if (state.isDraggingPoint) {
          state.isDraggingPoint = false;
          viewport.style.cursor = 'crosshair';
          if (state.draggedPointMoved) {
            recalibrateWithPoints(state.calibPoints, false);
          }
        }
      });

      // Canvas click for calibration mode or dart tip/tail
      viewport.addEventListener('click', async (e) => {
        if (e.button !== 0 || state.isPanning || state.justPanned || !state.imageLoaded) return;
        if (state.justHandledPoint) {
          state.justHandledPoint = false;
          return;
        }
        const pt = getImagePoint(e);
        if (pt.x < 0 || pt.x >= imgCanvas.width || pt.y < 0 || pt.y >= imgCanvas.height) return;

        // Wire-grid calibration mode
        if (state.calibMode) {
          state.calibClicks.push([pt.x, pt.y]);
          const stepIdx = state.calibClicks.length;
          if (stepIdx >= 4) {
            document.getElementById('btn-apply-calib-early').style.display = 'inline-block';
          }
          if (stepIdx < 24) {
            const nextStep = WIRE_GRID_STEPS[stepIdx];
            document.getElementById('mode-hint').innerHTML = `<strong style="color:#22c55e;">${nextStep.name}</strong>`;
            renderOverlay();
          } else {
            await applyWireGridCalibration();
          }
          return;
        }

        if (!state.pendingTip) {
          // Click 1: Dart tip
          state.pendingTip = [pt.x, pt.y];
          document.getElementById('mode-hint').innerHTML = '<strong>Tip set!</strong> Now click dart tail/flight (or click same point if unknown)';
          renderOverlay();
        } else {
          // Click 2: Dart tail
          const tip = state.pendingTip;
          const tail = [pt.x, pt.y];
          state.pendingTip = null;
          document.getElementById('mode-hint').innerHTML = 'Click 1: Dart Tip &bull; Click 2: Dart Tail';

          // Compute score from backend
          const scRes = await fetch(`/api/score?path=${encodeURIComponent(state.currentImagePath)}&x=${tip[0]}&y=${tip[1]}`);
          const scoreData = await scRes.json();

          state.darts.push({
            tip_px: tip,
            tail_px: tail,
            label: scoreData.label,
            sector: scoreData.sector,
            multiplier: scoreData.multiplier,
            score: scoreData.score
          });

          renderOverlay();
          renderDartsList();
        }
      });

      // Loupe mousemove
      viewport.addEventListener('mouseenter', () => loupe.style.display = 'block');
      viewport.addEventListener('mouseleave', () => loupe.style.display = 'none');

      // Buttons
      document.getElementById('chk-verified').addEventListener('change', (e) => {
        state.verified = e.target.checked;
        updateVerifiedBadge();
      });

      document.getElementById('btn-prev').addEventListener('click', prevImage);
      document.getElementById('btn-next').addEventListener('click', nextImage);
      document.getElementById('btn-save').addEventListener('click', saveAndNext);

      document.getElementById('btn-wire-grid-calib').addEventListener('click', () => {
        state.calibMode = true;
        state.calibClicks = [];
        state.pendingTip = null;
        document.getElementById('btn-apply-calib-early').style.display = 'none';
        const first = WIRE_GRID_STEPS[0];
        document.getElementById('mode-hint').innerHTML = `<strong style="color:#22c55e;">${first.name}</strong>`;
        renderOverlay();
      });

      document.getElementById('btn-apply-calib-early').addEventListener('click', async () => {
        await applyWireGridCalibration();
      });

      document.getElementById('btn-refit-wireframe').addEventListener('click', async () => {
        await recalibrateWithPoints(state.calibPoints, false);
        showToast('Wireframe re-fitted from current points!');
      });

      document.getElementById('btn-save-wireframe').addEventListener('click', async () => {
        await recalibrateWithPoints(state.calibPoints, true);
      });

      document.getElementById('btn-reset-calib-points').addEventListener('click', async () => {
        if (state.initialCalibPoints && state.initialCalibPoints.length > 0) {
          state.calibPoints = JSON.parse(JSON.stringify(state.initialCalibPoints));
          state.selectedPointIdx = null;
          updatePointInfo();
          await recalibrateWithPoints(state.calibPoints, false);
          showToast('Points reset to initial positions.');
        }
      });

      document.getElementById('toggle-calib-points').addEventListener('change', (e) => {
        state.showCalibPoints = e.target.checked;
        renderOverlay();
      });

      document.getElementById('btn-add-dart').addEventListener('click', () => {
        state.calibMode = false;
        document.getElementById('btn-apply-calib-early').style.display = 'none';
        document.getElementById('mode-hint').innerHTML = '<span style="color:#38bdf8;">Click on image to place Tip</span>';
      });

      document.getElementById('btn-undo-click').addEventListener('click', () => {
        if (state.calibMode && state.calibClicks.length > 0) {
          state.calibClicks.pop();
          if (state.calibClicks.length < 4) {
            document.getElementById('btn-apply-calib-early').style.display = 'none';
          }
          const stepIdx = state.calibClicks.length;
          const nextStep = WIRE_GRID_STEPS[stepIdx];
          document.getElementById('mode-hint').innerHTML = `<strong style="color:#22c55e;">${nextStep.name}</strong>`;
          renderOverlay();
        } else if (state.pendingTip) {
          state.pendingTip = null;
          document.getElementById('mode-hint').innerHTML = 'Click 1: Dart Tip &bull; Click 2: Dart Tail';
          renderOverlay();
        } else if (state.darts.length > 0) {
          state.darts.pop();
          renderOverlay();
          renderDartsList();
        }
      });

      document.getElementById('btn-clear-darts').addEventListener('click', () => {
        if (confirm('Clear all darts for this image?')) {
          state.darts = [];
          state.pendingTip = null;
          renderOverlay();
          renderDartsList();
        }
      });

      // Keyboard navigation
      window.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT') return;

        if (state.activeTab === 'annotator') {
          if (state.selectedPointIdx !== null && state.calibPoints && state.calibPoints[state.selectedPointIdx]) {
            const idx = state.selectedPointIdx;
            const pt = state.calibPoints[idx];
            const step = e.shiftKey ? 5 : 1;
            let handled = false;
            if (e.key === 'ArrowLeft') { pt[0] -= step; handled = true; }
            else if (e.key === 'ArrowRight') { pt[0] += step; handled = true; }
            else if (e.key === 'ArrowUp') { pt[1] -= step; handled = true; }
            else if (e.key === 'ArrowDown') { pt[1] += step; handled = true; }
            else if (e.key === 'Escape') {
              state.selectedPointIdx = null;
              updatePointInfo();
              renderOverlay();
              return;
            }

            if (handled) {
              e.preventDefault();
              updatePointInfo();
              renderOverlay();
              clearTimeout(state.refitTimeout);
              state.refitTimeout = setTimeout(() => {
                recalibrateWithPoints(state.calibPoints, false);
              }, 120);
              return;
            }
          }

          if (e.key === ' ' || e.key === 'Enter') {
            e.preventDefault();
            saveAndNext();
          } else if (e.key === 'n' || e.key === 'N') {
            nextImage();
          } else if (e.key === 'p' || e.key === 'P') {
            prevImage();
          } else if (e.key === 'r' || e.key === 'R') {
            resetTransform();
          } else if (e.key === 'o' || e.key === 'O') {
            state.showWireframe = !state.showWireframe;
            renderOverlay();
          } else if (e.key === 'z' || e.key === 'Z') {
            document.getElementById('btn-undo-click').click();
          } else if (e.key === 'v' || e.key === 'V') {
            const chk = document.getElementById('chk-verified');
            chk.checked = !chk.checked;
            state.verified = chk.checked;
            updateVerifiedBadge();
          }
        }
      });
    }

    function getImagePoint(e) {
      const rect = viewport.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      const imgX = (mouseX - state.panX) / state.scale;
      const imgY = (mouseY - state.panY) / state.scale;
      return { x: Math.round(imgX), y: Math.round(imgY) };
    }

    function updateLoupe(e) {
      if (!state.imageLoaded) return;
      const pt = getImagePoint(e);
      loupeCoord.textContent = `X: ${pt.x}, Y: ${pt.y}`;

      const s = 40;
      const sx = pt.x - s / 2;
      const sy = pt.y - s / 2;

      loupeCtx.imageSmoothingEnabled = false;
      loupeCtx.clearRect(0, 0, 180, 180);
      loupeCtx.drawImage(imgCanvas, sx, sy, s, s, 0, 0, 180, 180);

      loupeCtx.strokeStyle = '#38bdf8';
      loupeCtx.lineWidth = 1.5;
      loupeCtx.beginPath();
      loupeCtx.moveTo(90, 0); loupeCtx.lineTo(90, 180);
      loupeCtx.moveTo(0, 90); loupeCtx.lineTo(180, 90);
      loupeCtx.stroke();
    }

    function prevImage() {
      const sel = document.getElementById('image-select');
      if (sel.selectedIndex > 0) {
        sel.selectedIndex--;
        loadImage(sel.value);
      }
    }

    function nextImage() {
      const sel = document.getElementById('image-select');
      if (sel.selectedIndex < sel.options.length - 1) {
        sel.selectedIndex++;
        loadImage(sel.value);
      }
    }

    async function saveAndNext() {
      let tot = 0;
      state.darts.forEach(d => tot += (d.score || 0));

      const payload = {
        image_path: state.currentImagePath,
        calibration_points_px: state.calibPoints && state.calibPoints.length === 4 ? state.calibPoints : (state.calibration ? state.calibration.calib_points : null),
        wire_grid_points_px: state.calibPoints && state.calibPoints.length >= 4 ? state.calibPoints : null,
        darts: state.darts,
        total_score: tot,
        verified: state.verified
      };

      try {
        const res = await fetch('/api/save', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const ret = await res.json();
        state.dataset = ret.dataset;
        populateSelect();
        showToast('Saved annotations successfully!');
        nextImage();
      } catch (err) {
        alert('Save failed: ' + err);
      }
    }

    function showToast(msg) {
      const t = document.getElementById('toast');
      t.textContent = msg;
      t.className = 'toast show';
      setTimeout(() => t.className = 'toast', 2500);
    }

    window.addEventListener('DOMContentLoaded', init);
  </script>
</body>
</html>
"""


def _get_wireframe_data(H: np.ndarray, rings: dict | None = None) -> list[dict]:
    """Generate wireframe lines for web overlay, using fitted paint bands when available."""
    wires = []
    angles = np.linspace(0, 360, 361, endpoint=True)
    for r_name in ["bull", "outer_bull", "treble_inner", "treble_outer", "double_inner", "double_outer"]:
        if rings and r_name in rings:
            val = rings[r_name]
            c = val.contour if hasattr(val, "contour") else val
            step = max(1, len(c) // 360)
            pts_px = c[::step].reshape(-1, 2).tolist()
            wires.append({"type": "circle", "name": r_name, "points": pts_px})
        else:
            r = RADII[r_name]
            pts_mm = [polar(r, a) for a in angles]
            pts_px = project(H, pts_mm).tolist()
            wires.append({"type": "circle", "name": r_name, "points": pts_px})

    for i in range(20):
        a = 9.0 + 18.0 * i
        p_in = polar(RADII["outer_bull"], a)
        p_out = polar(RADII["double_outer"], a)
        pts_px = project(H, [p_in, p_out]).tolist()
        wires.append({"type": "line", "sector": SECTORS[i], "points": pts_px})

    return wires


def calibrate_from_wire_grid(
    pts: list[list[float]] | np.ndarray,
) -> tuple[np.ndarray, dict, list[dict], list[float], list[float]]:
    """Accurately calibrate homography and scalar ring radii via Joint Projective Alternating Least Squares (H, R_k).

    Models the dartboard as an 8-DoF planar projective homography H with 6 exact concentric model circles
    (R_do = 170.0 mm, R_di, R_to, R_ti, R_ob, R_bull) and 20 straight radial spokes.
    Alternating Least Squares fits H over all clicked handles while jointly solving the scalar ring radii.
    """
    pts = np.float32(pts)
    n = len(pts)

    cardinal_angles = [9.0, 99.0, 189.0, 279.0]
    initial_radii = [170.0, 159.75, 106.6, 96.7, 16.1, 6.9]
    radii = list(initial_radii)

    ring_indices = {
        0: [0, 5, 10, 15],  # double_outer
        1: [1, 6, 11, 16],  # double_inner
        2: [2, 7, 12, 17],  # treble_outer
        3: [3, 8, 13, 18],  # treble_inner
        4: [4, 9, 14, 19],  # outer_bull
        5: [20, 21, 22, 23],  # bull
    }

    if n < 4:
        H = np.eye(3, dtype=np.float32)
    elif n < 16:
        H = compute_homography(pts[:4])
    else:
        for _ in range(8):
            model_pts = []
            for s in range(4):
                ang = cardinal_angles[s]
                for k in range(5):
                    model_pts.append(polar(radii[k], ang))
            for s in range(4):
                ang = cardinal_angles[s]
                model_pts.append(polar(radii[5], ang))
            model_pts = np.float32(model_pts[:n])

            H, _ = cv2.findHomography(model_pts, pts[:n], 0)
            if H is None:
                H = compute_homography([pts[0], pts[5], pts[10], pts[15]])
                break

            H_inv = np.linalg.inv(H)
            m_pts = project(H_inv, pts[:n])

            # Fix scale by setting mean radius of outer double points to 170.0 mm
            r_outer_mean = float(np.mean(np.linalg.norm(m_pts[ring_indices[0]], axis=1)))
            if r_outer_mean > 1e-3:
                scale_factor = 170.0 / r_outer_mean
                m_pts = m_pts * scale_factor
                H = H @ np.diag([1.0 / scale_factor, 1.0 / scale_factor, 1.0])

            radii[0] = 170.0
            for k in range(1, 6):
                idxs = [idx for idx in ring_indices[k] if idx < n]
                if idxs:
                    radii[k] = float(np.mean(np.linalg.norm(m_pts[idxs], axis=1)))

    ring_names = ["double_outer", "double_inner", "treble_outer", "treble_inner", "outer_bull", "bull"]
    angles = np.linspace(0, 360, 361, endpoint=True)
    rings_contours: dict[str, np.ndarray] = {}
    wires: list[dict] = []

    for k in (5, 4, 3, 2, 1, 0):
        r_name = ring_names[k]
        r_val = radii[k]
        pts_m = [polar(r_val, a) for a in angles]
        curve_px = project(H, pts_m)
        rings_contours[r_name] = curve_px
        wires.append({"type": "circle", "name": r_name, "points": curve_px.tolist()})

    for i in range(20):
        a = 9.0 + 18.0 * i
        p_in = polar(radii[4], a)
        p_out = polar(radii[0], a)
        pts_line = project(H, [p_in, p_out]).tolist()
        wires.append({"type": "line", "sector": SECTORS[i], "points": pts_line})

    center_px = project(H, [[0, 0]])[0].tolist()
    return H, rings_contours, wires, center_px, radii


def reload_pipeline_modules() -> None:
    """Reload all core dartcv pipeline modules so runs always execute latest code from disk."""
    import importlib
    import sys
    modules = [
        "dartcv.types",
        "dartcv.geometry",
        "dartcv.rings",
        "dartcv.darts",
        "dartcv.detect",
        "dartcv.overlay",
        "dartcv.diagnostics",
        "dartcv.evaluate",
    ]
    for mod_name in modules:
        if mod_name in sys.modules:
            try:
                importlib.reload(sys.modules[mod_name])
            except Exception as e:
                print(f"Warning: could not reload {mod_name}: {e}")


def render_analysis_composite(img: np.ndarray, analysis: BoardAnalysis) -> np.ndarray:
    """Render a side-by-side composite visualization of live board analysis."""
    h, w = img.shape[:2]
    scale = 1000.0 / h
    disp_w = int(w * scale)
    disp_h = 1000
    left = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    if analysis.board:
        S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float32)
        left = draw_overlay(left, S @ analysis.board.H)
        topdown = warp_topdown(img, analysis.board.H)
        topdown = cv2.resize(topdown, (disp_h, disp_h), interpolation=cv2.INTER_AREA)
    else:
        topdown = np.zeros((disp_h, disp_h, 3), dtype=np.uint8)

    for idx, d in enumerate(analysis.darts):
        tip = (int(d.tip_px[0] * scale), int(d.tip_px[1] * scale))
        if d.tail_px is not None:
            tail = (int(d.tail_px[0] * scale), int(d.tail_px[1] * scale))
            cv2.line(left, tail, tip, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.circle(left, tail, 6, (0, 200, 255), -1, cv2.LINE_AA)
        cv2.circle(left, tip, 9, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(left, tip, 13, (255, 255, 255), 2, cv2.LINE_AA)
        lbl_text = f"Dart {idx+1}: {d.label} ({d.score} pts)"
        cv2.putText(left, lbl_text, (tip[0] + 16, tip[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    # Info banner on top of left panel
    cv2.rectangle(left, (10, 10), (360, 50), (15, 23, 42), -1)
    cv2.putText(left, f"Score: {analysis.total_score} pts ({len(analysis.darts)} darts)", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    return np.hstack([left, topdown])


def export_standalone_dashboard(metrics: dict, out_path: Path) -> None:
    """Export standalone HTML dashboard with embedded evaluation metrics."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = HTML_TEMPLATE.replace("/*METRICS_JSON*/ null", json.dumps(metrics))
    out_path.write_text(html, encoding="utf-8")
    print(f"Exported standalone dashboard to {out_path}")


class AnnotationServer(http.server.SimpleHTTPRequestHandler):
    annotations_path: Path = DEFAULT_ANNOTATIONS
    calib_cache: dict = {}
    live_analysis_cache: dict = {}

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)

        if parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        if parsed.path == "/api/dataset":
            if self.annotations_path.is_file():
                with open(self.annotations_path) as f:
                    data = json.load(f)
            else:
                data = {"version": 1, "starter_15": [], "images": {}}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        if parsed.path == "/api/image":
            path_str = qs.get("path", [""])[0]
            img_path = Path(path_str)
            if img_path.is_file():
                self.send_response(200)
                content_type = "image/jpeg" if img_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
                self.send_header("Content-Type", content_type)
                self.end_headers()
                with open(img_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
            return

        if parsed.path == "/api/calibration":
            path_str = qs.get("path", [""])[0]
            img_path = Path(path_str)
            if path_str in self.calib_cache:
                cal_data = self.calib_cache[path_str]
            elif img_path.is_file():
                entry_wg = None
                if self.annotations_path.is_file():
                    with open(self.annotations_path) as f:
                        gt_all = json.load(f)
                        entry_wg = gt_all.get("images", {}).get(path_str, {}).get("wire_grid_points_px")

                if entry_wg and len(entry_wg) >= 4:
                    H, rings_dict, wires, center_px, radii = calibrate_from_wire_grid(entry_wg)
                    cal_data = {
                        "ecc_score": 1.0,
                        "rotation_deg": 0.0,
                        "center": center_px,
                        "wires": wires,
                        "H": H.tolist(),
                        "radii": radii,
                        "rings_count": len(rings_dict),
                        "rings_contours": rings_dict,
                        "is_manual": True,
                        "wire_grid_points": entry_wg,
                        "calib_points": project(H, calibration_model_points()).tolist(),
                    }
                else:
                    img = cv2.imread(str(img_path))
                    det = detect_board(img)
                    wires = _get_wireframe_data(det.H, det.rings)
                    center_px = project(det.H, [[0, 0]])[0].tolist()
                    rings_dict = {k: v.contour for k, v in det.rings.items()} if det.rings else {}
                    cal_data = {
                        "ecc_score": float(det.ecc_score) if det.ecc_score is not None else 0.0,
                        "rotation_deg": float(det.rotation_deg),
                        "center": center_px,
                        "wires": wires,
                        "H": det.H.tolist(),
                        "rings_count": len(det.rings) if det.rings else 0,
                        "rings_contours": rings_dict,
                    }
                self.calib_cache[path_str] = cal_data
            else:
                cal_data = {"error": "Image not found"}

            resp_data = {k: v for k, v in cal_data.items() if k != "rings_contours"}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp_data).encode("utf-8"))
            return

        if parsed.path == "/api/score":
            path_str = qs.get("path", [""])[0]
            x = float(qs.get("x", [0])[0])
            y = float(qs.get("y", [0])[0])
            cal_data = self.calib_cache.get(path_str)

            if cal_data and "radii" in cal_data:
                H = np.array(cal_data["H"], dtype=np.float32)
                rads = cal_data["radii"]
                pts_m = project(np.linalg.inv(H), [[x, y]])[0]
                r = math.hypot(float(pts_m[0]), float(pts_m[1]))
                angle = (math.degrees(math.atan2(float(pts_m[0]), -float(pts_m[1]))) + 9.0) % 360.0
                sector = SECTORS[int(angle // 18) % 20]
                if r <= rads[5]:
                    score, sector, multiplier, label = 50, 50, 2, "DBULL"
                elif r <= rads[4]:
                    score, sector, multiplier, label = 25, 25, 1, "BULL"
                elif r > rads[0]:
                    score, sector, multiplier, label = 0, 0, 0, "MISS"
                elif r > rads[1]:
                    score, sector, multiplier, label = 2 * sector, sector, 2, f"D{sector}"
                elif rads[3] < r <= rads[2]:
                    score, sector, multiplier, label = 3 * sector, sector, 3, f"T{sector}"
                else:
                    score, sector, multiplier, label = sector, sector, 1, f"S{sector}"
            else:
                rings = None
                if not cal_data:
                    img_path = Path(path_str)
                    if img_path.is_file():
                        img = cv2.imread(str(img_path))
                        det = detect_board(img)
                        H = det.H
                        rings = {k: v.contour for k, v in det.rings.items()} if det.rings else None
                    else:
                        H = np.eye(3)
                else:
                    H = np.array(cal_data["H"], dtype=np.float32)
                    rings = cal_data.get("rings_contours")

                score, sector, multiplier, label = score_point_detailed(H, [x, y], rings=rings)

            res = {
                "score": int(score),
                "sector": int(sector),
                "multiplier": int(multiplier),
                "label": str(label),
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        # DASHBOARD API ENDPOINTS
        if parsed.path == "/api/eval_metrics":
            metrics_file = DEFAULT_OUT / "metrics.json"
            if not metrics_file.is_file():
                from .evaluate import evaluate_dataset
                metrics = evaluate_dataset(self.annotations_path, DEFAULT_OUT, save_composites=True)
            else:
                with open(metrics_file) as f:
                    metrics = json.load(f)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(metrics).encode("utf-8"))
            return

        if parsed.path == "/api/unannotated_images":
            with open(self.annotations_path) as f:
                gt = json.load(f)
            starter = set(gt.get("starter_15", []))

            categories = {"raw_board": [], "single_dart": [], "multi_dart": []}
            for cat in ("raw_board", "single_dart", "multi_dart"):
                p = Path("data/images") / cat
                if p.is_dir():
                    for f in sorted(p.glob("*.jpg")):
                        sp = str(f)
                        if sp not in starter:
                            categories[cat].append(sp)

            starter_9 = []
            for cat in ("raw_board", "single_dart", "multi_dart"):
                starter_9.extend(categories[cat][:3])

            resp = {
                "total": sum(len(v) for v in categories.values()),
                "categories": categories,
                "starter_9": starter_9,
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        if parsed.path == "/api/analyze_live":
            path_str = qs.get("path", [""])[0]
            img_path = Path(path_str)
            if img_path.is_file():
                img = cv2.imread(str(img_path))
                if img is not None:
                    reload_pipeline_modules()
                    import dartcv.detect as detect_mod
                    analysis = detect_mod.analyze_image(img, diagnostics=True)
                    self.live_analysis_cache[path_str] = analysis
                    wires = _get_wireframe_data(analysis.board.H, analysis.board.rings) if analysis.board else []
                    center = project(analysis.board.H, [[0, 0]])[0].tolist() if analysis.board else [0, 0]
                    darts_data = [
                        {
                            "tip_px": [round(float(d.tip_px[0]), 1), round(float(d.tip_px[1]), 1)],
                            "tail_px": [round(float(d.tail_px[0]), 1), round(float(d.tail_px[1]), 1)] if d.tail_px is not None else None,
                            "label": str(d.label),
                            "score": int(d.score),
                            "confidence": round(float(d.confidence), 2),
                        }
                        for d in analysis.darts
                    ]
                    res = {
                        "path": path_str,
                        "board": {
                            "center": center,
                            "ecc_score": round(float(analysis.board.ecc_score), 4) if analysis.board and analysis.board.ecc_score is not None else 0.0,
                            "rotation_deg": round(float(analysis.board.rotation_deg), 2) if analysis.board else 0.0,
                            "wires": wires,
                        } if analysis.board else None,
                        "darts": darts_data,
                        "total_score": int(analysis.total_score),
                        "timings_ms": {
                            "board_ms": round(float(analysis.timings_ms.get("board_ms", 0.0)), 1),
                            "darts_ms": round(float(analysis.timings_ms.get("darts_ms", 0.0)), 1),
                            "total_ms": round(float(analysis.timings_ms.get("total_ms", 0.0)), 1),
                        },
                    }
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(res).encode("utf-8"))
                    return
            self.send_response(404)
            self.end_headers()
            return

        if parsed.path == "/api/render_analysis":
            path_str = qs.get("path", [""])[0]
            img_path = Path(path_str)
            if img_path.is_file():
                img = cv2.imread(str(img_path))
                if img is not None:
                    analysis = self.live_analysis_cache.get(path_str)
                    if analysis is None:
                        reload_pipeline_modules()
                        import dartcv.detect as detect_mod
                        analysis = detect_mod.analyze_image(img, diagnostics=True)
                        self.live_analysis_cache[path_str] = analysis
                    comp = render_analysis_composite(img, analysis)
                    ret, jpeg = cv2.imencode(".jpg", comp, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                    if ret:
                        self.send_response(200)
                        self.send_header("Content-Type", "image/jpeg")
                        self.end_headers()
                        self.wfile.write(jpeg.tobytes())
                        return
            self.send_response(404)
            self.end_headers()
            return

        if parsed.path == "/api/composite":
            file_name = qs.get("file", [""])[0]
            comp_path = DEFAULT_OUT / file_name
            if comp_path.is_file():
                self.send_response(200)
                content_type = "image/png" if comp_path.suffix.lower() == ".png" else "image/jpeg"
                self.send_header("Content-Type", content_type)
                self.end_headers()
                with open(comp_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            self.send_response(404)
            self.end_headers()
            return

        super().do_GET()

    def do_POST(self):
        if self.path == "/api/run_eval":
            reload_pipeline_modules()
            self.live_analysis_cache.clear()
            import dartcv.evaluate as eval_mod
            metrics = eval_mod.evaluate_dataset(self.annotations_path, DEFAULT_OUT, save_composites=True)
            export_standalone_dashboard(metrics, DEFAULT_OUT / "dashboard.html")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(metrics).encode("utf-8"))
            return

        if self.path == "/api/recalibrate_wire_grid":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            body = json.loads(post_data.decode("utf-8"))
            path_str = body["path"]
            pts = body["points"]
            should_save = bool(body.get("save", True))

            H, rings_dict, wires, center_px, radii = calibrate_from_wire_grid(pts)
            calib_4pts = project(H, calibration_model_points()).tolist()

            cal_data = {
                "ecc_score": 1.0,
                "rotation_deg": 0.0,
                "center": center_px,
                "wires": wires,
                "H": H.tolist(),
                "radii": radii,
                "rings_count": len(rings_dict),
                "is_manual": True,
                "calib_points": calib_4pts,
                "wire_grid_points": pts,
                "rings_contours": rings_dict,
            }
            self.calib_cache[path_str] = cal_data

            if should_save:
                with open(self.annotations_path) as f:
                    dataset = json.load(f)
                if path_str in dataset["images"]:
                    dataset["images"][path_str]["calibration_points_px"] = calib_4pts
                    dataset["images"][path_str]["wire_grid_points_px"] = pts
                with open(self.annotations_path, "w") as f:
                    json.dump(dataset, f, indent=2)

            resp_data = {k: v for k, v in cal_data.items() if k != "rings_contours"}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp_data).encode("utf-8"))
            return

        if self.path == "/api/save":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            entry = json.loads(post_data.decode("utf-8"))

            with open(self.annotations_path) as f:
                data = json.load(f)

            img_path = entry["image_path"]
            if "images" not in data:
                data["images"] = {}

            existing = data["images"].get(img_path, {})
            existing_cal = existing.get("calibration_points_px")
            existing_wg = existing.get("wire_grid_points_px")

            data["images"][img_path] = {
                "image_path": img_path,
                "calibration_points_px": entry.get("calibration_points_px", existing_cal),
                "wire_grid_points_px": entry.get("wire_grid_points_px", existing_wg),
                "darts": entry.get("darts", []),
                "total_score": entry.get("total_score", 0),
                "verified": bool(entry.get("verified", False)),
            }

            with open(self.annotations_path, "w") as f:
                json.dump(data, f, indent=2)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "dataset": data}).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()


def run_web_annotator(annotations_path: Path, port: int = 8080) -> None:
    AnnotationServer.annotations_path = annotations_path
    socketserver.TCPServer.allow_reuse_address = True
    server_address = ("", port)
    with socketserver.TCPServer(server_address, AnnotationServer) as httpd:
        print(f"DartCV Annotator web server running at http://localhost:{port}/")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down annotator server.")


def run_opencv_annotator(image_path: Path, annotations_path: Path) -> None:
    """Fallback OpenCV desktop GUI annotator when display is connected."""
    img = cv2.imread(str(image_path))
    if img is None:
        sys.exit(f"Cannot load image {image_path}")

    det = detect_board(img)
    H = det.H
    win = f"dartcv-annotate: {image_path.name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1200, 900)
    print(f"Opened {image_path.name} in OpenCV window. Left click to add dart, Backspace undo, Enter save.")
    cv2.destroyAllWindows()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="dartcv-annotate", description="Interactive dartboard ground-truth annotator & telemetry dashboard")
    parser.add_argument("image", nargs="?", default=None, help="Specific image to annotate")
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS), help="Annotations JSON file")
    parser.add_argument("--port", type=int, default=8080, help="Web server port (default: 8080)")
    parser.add_argument("--gui", action="store_true", help="Force OpenCV GUI (requires DISPLAY)")
    parser.add_argument("--web", action="store_true", default=True, help="Run web-based annotator (default)")
    parser.add_argument("--export-html", default=None, help="Export standalone HTML file")
    parser.add_argument("--export-dashboard", default=None, help="Export standalone evaluation dashboard HTML")
    args = parser.parse_args(argv)

    ann_path = Path(args.annotations)
    if args.export_html:
        out_path = Path(args.export_html)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(HTML_TEMPLATE, encoding="utf-8")
        print(f"Exported annotator HTML to {out_path}")
        return

    if args.export_dashboard:
        from .evaluate import evaluate_dataset
        out_dash = Path(args.export_dashboard)
        metrics = evaluate_dataset(ann_path, DEFAULT_OUT)
        export_standalone_dashboard(metrics, out_dash)
        return

    has_display = bool(os.environ.get("DISPLAY"))
    if args.gui and has_display and args.image:
        run_opencv_annotator(Path(args.image), ann_path)
    else:
        run_web_annotator(ann_path, port=args.port)


if __name__ == "__main__":
    main()
