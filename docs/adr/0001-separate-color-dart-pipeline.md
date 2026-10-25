# Keep the color-based multi-dart pipeline separate

The color-based multi-dart recognizer is a distinct, incrementally developed
workflow, so it will have its own class, stage/result contracts, board constants,
and scoring implementation rather than extending the existing prototype or
sharing its scorer. This preserves independent replacement and testing of its
color-based stages; the deliberate cost is duplicated standard-board geometry
that must be kept consistent.
