"""Dart computer vision utilities."""

from .board import point_to_score, render_dartboard, render_opencv, render_svg


def main() -> None:
    """Small CLI entry point for the project."""
    print("dartcv: synthetic dartboard template utilities loaded.")
    print("Use dartcv.board.render_dartboard(), render_svg(), or render_opencv().")


__all__ = ["main", "point_to_score", "render_dartboard", "render_opencv", "render_svg"]
