import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dartcv.color_dart_inspection import plot_color_segmentation


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "examples").glob("*.ipynb"))


def test_scoring_notebook_model_examples():
    notebook = json.loads((ROOT / "examples/dart_scorer_pipeline.ipynb").read_text())
    namespace = {}
    for cell in notebook["cells"]:
        if cell.get("id") in ("imports", "stage1-checks"):
            exec("".join(cell["source"]), namespace)


@pytest.mark.parametrize("name", ["color_dart_pipeline_overview", "color_dart_segmentation_inspection"])
def test_segmentation_notebook_reports_stale_analysis(name):
    notebook = json.loads((ROOT / "examples" / f"{name}.ipynb").read_text())
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == "segmentation-inspection")
    old_result = SimpleNamespace(artifacts={
        "color_model_centers": np.zeros((4, 3)),
        "expected_field_color": np.zeros((2, 2), dtype=np.uint8),
    })
    with pytest.raises(RuntimeError, match="Restart.*Run All"):
        exec("".join(cell["source"]), {
            "analysis": SimpleNamespace(segmentation=old_result),
            "display": lambda figure: None,
            "plot_color_segmentation": plot_color_segmentation,
        })


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_runs_in_fresh_kernel(notebook, tmp_path):
    if notebook.stem != "dartboard_visualization" and not all(
        list((ROOT / "data/images" / folder).glob("*.jpg"))
        for folder in ("raw_board", "single_dart", "multi_dart")
    ):
        pytest.skip("Restore the local DVC photos to execute image notebooks.")
    for name in ("pyproject.toml", "src", "data"):
        (tmp_path / name).symlink_to(ROOT / name)
    examples = tmp_path / "examples"
    examples.mkdir()
    result = subprocess.run(
        [
            sys.executable, "-c",
            """
import json
import sys
from IPython.core.interactiveshell import InteractiveShell
import matplotlib.pyplot as plt

shell = InteractiveShell.instance()
shell.user_ns["display"] = lambda *args, **kwargs: None
with open(sys.argv[1]) as source:
    notebook = json.load(source)
for cell in notebook["cells"]:
    if cell["cell_type"] != "code":
        continue
    result = shell.run_cell("".join(cell["source"]))
    if result.error_before_exec or result.error_in_exec:
        raise RuntimeError(f"Notebook cell {cell.get('id')} failed") from (
            result.error_before_exec or result.error_in_exec
        )
    plt.close("all")
""",
            str(notebook),
        ],
        cwd=examples,
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
