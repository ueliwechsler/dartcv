# Architecture documentation template

Use this structure for the living [project architecture](../architecture.md).
Replace placeholders with verified facts and relative source links. Separate
current behavior from proposed designs; omit no section without an explanation.
Do not copy this template for every small code change: maintain the living document.

## Purpose and scope

Describe what the system currently does and what it explicitly does not do.

## Components and responsibilities

| Component / source link | Responsibility | Interfaces and dependencies |
| --- | --- | --- |
| <module> | <owned behavior> | <entry points, inputs/outputs, dependencies> |

## Runtime and data flow

Describe entry points and alternative execution paths. Include a diagram when
useful; distinguish call dependencies from data flow. Identify side effects.

## Contracts and invariants

Specify coordinate systems, units, transform directions, array/image conventions,
returned data, ownership, and public compatibility requirements.

## Design decisions and trade-offs

Explain important choices, constraints, alternatives where known, and consequences.
Do not invent historical rationale; label inferred explanations or open decisions.

## Dependencies, data, and artifacts

Link dependency configuration; explain runtime/tooling roles, data ownership,
generated outputs, persistence, and secret-handling boundaries.

## Validation and limitations

Link tests and describe what they establish, required checks, known failure modes,
and unsupported behavior. Keep measured performance separate from expectations.

## Related documentation and maintenance

Link user documentation and detailed technical documents. State what changes
require an update here and which details belong in those other documents.