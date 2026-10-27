"""Shared data structures for board pose, dart detection, and end-to-end analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .detect import Detection


@dataclass
class DartDetection:
    tip_px: np.ndarray          # (2,) float64 in original image pixels (x, y)
    tail_px: np.ndarray         # (2,) float64 tail/flight end along dart axis in original pixels
    tip_mm: np.ndarray          # (2,) float64 in canonical board mm (origin at bull, 20 up)
    sector: int                 # 1..20, 25 (outer bull), 50 (inner bull), or 0 (miss)
    multiplier: int             # 0, 1, 2, or 3
    score: int                  # 0..60
    label: str                  # "T20", "D16", "S18", "BULL", "DBULL", "MISS"
    confidence: float           # 0.0..1.0 detection confidence


@dataclass
class DartDiagnostics:
    topdown_anomaly: np.ndarray | None = None      # (H, W) uint8 or float32 anomaly map
    topdown_ridges: np.ndarray | None = None       # (H, W) uint8 or float32 wire-suppressed ridge map
    topdown_expected: np.ndarray | None = None     # (H, W, 3) uint8 reconstructed clean board
    strip_crops: list[np.ndarray] = field(default_factory=list)


@dataclass
class BoardAnalysis:
    board: Detection
    darts: list[DartDetection] = field(default_factory=list)
    total_score: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)
    dart_diagnostics: DartDiagnostics | None = None
