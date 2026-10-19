# Repository instructions

## Documentation

For every code change, assess documentation impact and update affected docs and
docstrings **in the same change**. Read only the documents relevant to the task:

- Structure, interfaces, dependencies, contracts: [architecture](docs/architecture.md).
- CV algorithms, geometry, scoring, diagnostics, fallbacks: [pipeline](docs/cv-pipeline.md).
- CLI, outputs, setup, data workflows: [README](README.md).
- Rendering/example APIs: [notebook](examples/dartboard_visualization.ipynb).
- Architecture document structure: [template](docs/templates/architecture.md);
  keep it reusable, updating it only when the documentation structure changes.

Document implemented behavior and limitations; distinguish measurements from model
projections. ECC scores/fit residuals are not ground-truth accuracy. Percentage
accuracy requires a defined metric and evaluated dataset. Avoid cosmetic doc edits.

## Validation and handoff

- Run relevant tests: `uv run --group dev pytest` (focused subsets allowed).
  Docs-only changes: check links and source consistency; no tests required.
- Real-image checks: **one explicitly selected image** unless a batch is requested;
  synthetic tests are unrestricted. Do not change CLI batch behavior for this rule.
- Keep secrets local; do not modify/upload DVC data incidentally.
- Finish with docs updated (or why none were needed), validation results, and
  checks not run or remaining uncertainties.