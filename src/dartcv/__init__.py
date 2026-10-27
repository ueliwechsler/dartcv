"""Dart computer vision utilities."""

from .board import point_to_score, render_dartboard, render_opencv, render_svg
from .detect import Detection, analyze_image, detect_board
from .types import BoardAnalysis, DartDetection, DartDiagnostics


def main() -> None:
    """Small CLI entry point for the project."""
    print("dartcv: synthetic dartboard template utilities loaded.")
    print("Use dartcv.board.render_dartboard(), render_svg(), or render_opencv().")


__all__ = [
    "BoardAnalysis",
    "DartDetection",
    "DartDiagnostics",
    "Detection",
    "analyze_image",
    "detect_board",
    "main",
    "point_to_score",
    "render_dartboard",
    "render_opencv",
    "render_svg",
]
