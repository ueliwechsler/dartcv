"""Skeleton pipeline for automatic dart detection on a dartboard.

This module intentionally captures the structure of the colour-based dartboard
pipeline described in the project PDF without trying to solve the full algorithm
in one step. The goal is to keep every stage modular, easy to replace, and easy
to inspect while a later implementation fills in the actual CV logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

import cv2
import numpy as np


@dataclass
class PipelineStepResult:
    """One stage of the pipeline with captured I/O and timing."""

    name: str
    elapsed_ms: float = 0.0
    input_image: np.ndarray | None = None
    output: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return f"{self.name}: {self.elapsed_ms:.2f} ms"


@dataclass
class PipelineResult:
    """Entire pipeline run, including diagnostic images and stage timings."""

    image: np.ndarray
    board_mask: np.ndarray | None = None
    dart_mask: np.ndarray | None = None
    darts: list[dict[str, Any]] = field(default_factory=list)
    steps: list[PipelineStepResult] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def debug_images(self) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        for step in self.steps:
            if isinstance(step.output, np.ndarray):
                images[step.name] = step.output
        return images

    def summary(self) -> str:
        return "\n".join(step.summary() for step in self.steps)


class DartboardDartRecognitionPipeline:
    """Modular skeleton for the dart-detection pipeline.

    The default stage order follows the PDF's classical approach:
    1. board mask
    2. HSV cone features
    3. robust colour-classification and outlier mask
    4. edge-preserving pre-filter
    5. opening + local regrow
    6. minimum connected-component filter
    7. hole fill + GrabCut refinement

    Each stage is a callable with the same lightweight interface and stores its
    input/output arrays in `PipelineResult.steps`, so the object can be inspected
    at every step. This makes it easy to swap out a single stage later without
    refactoring the rest of the pipeline.
    """

    def __init__(
        self,
        *,
        debug: bool = True,
        stages: list[tuple[str, Callable[[np.ndarray, dict[str, Any]], Any]]]
        | None = None,
    ) -> None:
        self.debug = debug
        self._stages = list(stages) if stages is not None else self._default_stages()
        self.latest_result: PipelineResult | None = None

    def _default_stages(
        self,
    ) -> list[tuple[str, Callable[[np.ndarray, dict[str, Any]], Any]]]:
        return [
            ("board_mask", self._board_mask_stage),
            ("hsv_features", self._hsv_feature_stage),
            ("colour_classification", self._colour_classification_stage),
            ("edge_preserving_filter", self._edge_preserving_filter_stage),
            ("morphology_opening", self._morphology_stage),
            ("minimum_blob_filter", self._minimum_blob_stage),
            ("grabcut_refinement", self._grabcut_stage),
        ]

    @staticmethod
    def _board_mask_stage(image: np.ndarray, state: dict[str, Any]) -> np.ndarray:
        """Return a board ROI mask.

        The default implementation is intentionally conservative: full-frame mask
        with a note that a production algorithm should use the board detector.
        """
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        state["board_mask"] = mask
        return mask

    @staticmethod
    def _hsv_feature_stage(image: np.ndarray, state: dict[str, Any]) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        state["hsv_features"] = hsv
        return hsv

    @staticmethod
    def _colour_classification_stage(
        image: np.ndarray, state: dict[str, Any]
    ) -> np.ndarray:
        hsv = state.get("hsv_features")
        if hsv is None:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Placeholder classifier: keep a compact non-zero mask so the pipeline is
        # runnable while the real colour model is added later.
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        if image.size:
            mask = np.clip(hsv[..., 2], 0, 255).astype(np.uint8)
        state["colour_mask"] = mask
        return mask

    @staticmethod
    def _edge_preserving_filter_stage(
        image: np.ndarray, state: dict[str, Any]
    ) -> np.ndarray:
        if state.get("colour_mask") is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            mask = state["colour_mask"]
        filtered = cv2.medianBlur(mask.astype(np.uint8), 5)
        state["filtered_mask"] = filtered
        return filtered

    @staticmethod
    def _morphology_stage(image: np.ndarray, state: dict[str, Any]) -> np.ndarray:
        mask = state.get("filtered_mask")
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        state["opened_mask"] = opened
        return opened

    @staticmethod
    def _minimum_blob_stage(image: np.ndarray, state: dict[str, Any]) -> np.ndarray:
        mask = state.get("opened_mask")
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num <= 1:
            result = mask
        else:
            keep = np.zeros_like(mask, dtype=np.uint8)
            for idx in range(1, num):
                if stats[idx, cv2.CC_STAT_AREA] >= 1500:
                    keep[labels == idx] = 255
            result = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        state["dart_seed"] = result
        return result

    @staticmethod
    def _grabcut_stage(image: np.ndarray, state: dict[str, Any]) -> np.ndarray:
        mask = state.get("dart_seed")
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        result = np.uint8(mask > 0) * 255
        state["dart_mask"] = result
        return result

    def register_stage(
        self,
        name: str,
        stage: Callable[[np.ndarray, dict[str, Any]], Any],
    ) -> None:
        self._stages.append((name, stage))

    def reset(self) -> None:
        self.latest_result = None

    def run(self, image: np.ndarray) -> PipelineResult:
        """Execute each configured stage and keep the input/output arrays."""
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be a BGR uint8 array with shape (H, W, 3)")
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        state: dict[str, Any] = {"image": image}
        steps: list[PipelineStepResult] = []
        result = PipelineResult(image=image.copy(), darts=[], notes=[])

        for name, stage in self._stages:
            start = perf_counter()
            output = stage(image, state)
            elapsed_ms = (perf_counter() - start) * 1000.0
            metadata: dict[str, Any] = {}
            if isinstance(output, dict):
                metadata = output.get("metadata", {})
                output_value = output.get("output")
            else:
                output_value = output
            if output_value is not None:
                output_value = np.asarray(output_value)
            step = PipelineStepResult(
                name=name,
                elapsed_ms=elapsed_ms,
                input_image=image.copy(),
                output=output_value,
                metadata=metadata,
            )
            steps.append(step)
            result.timings_ms[name] = elapsed_ms
            if self.debug:
                result.notes.append(step.summary())
            if isinstance(output_value, np.ndarray) and output_value.ndim >= 2:
                if name == "board_mask":
                    result.board_mask = output_value
                if name == "grabcut_refinement":
                    result.dart_mask = output_value

        result.steps = steps
        result.dart_mask = (
            result.dart_mask if result.dart_mask is not None else state.get("dart_mask")
        )
        result.board_mask = (
            result.board_mask
            if result.board_mask is not None
            else state.get("board_mask")
        )
        self.latest_result = result
        return result

    def stage_summary(self) -> str:
        if self.latest_result is None:
            return "No pipeline results yet. Call run(image)."
        return self.latest_result.summary()


__all__ = [
    "DartboardDartRecognitionPipeline",
    "PipelineResult",
    "PipelineStepResult",
]
