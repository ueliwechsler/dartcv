# Dartboard Recognition

This context describes how a board image is normalized, dart entry points are
estimated, and scoring outcomes are represented.

## Language

**Visible dart region**:
The portion of an individual dart visible in a photograph, excluding parts
hidden behind other objects.
_Avoid_: Complete dart silhouette

**Inferred dart silhouette**:
The estimated complete outline of an individual dart, including hidden parts;
it is distinct from the dart's visible regions.
_Avoid_: Visible dart mask, observed outline

**Dart tip**:
The physical point of a dart, whether or not it is embedded in a board.
_Avoid_: Board entry point when board contact is unknown

**Annotation ignore region**:
A region whose visible dart-versus-background identity cannot be reliably
determined and is excluded from segmentation evaluation.
_Avoid_: Background, uncertain dart part

**Uncertain dart part**:
A visible region known to belong to a dart whose flight-versus-shaft/barrel
identity is uncertain.
_Avoid_: Uncertain foreground, annotation ignore region

**Board entry point**:
The point where a dart enters the board surface and determines its score; it
may be observed directly or estimated from visible dart geometry.
_Avoid_: Dart center, box center

**Normalized board view**:
A perspective-corrected view of the standard dartboard with a transform back to
the source image.
_Avoid_: Board crop

**20-sector orientation**:
The alignment that identifies which scoring wedge is numbered 20; alternating
ring colors alone do not uniquely determine it.
_Avoid_: Board rotation, upright board

**Number-ring view**:
A perspective-corrected view that includes the outer numbered labels, beyond
the scoring surface. Unobserved areas are distinguished from visible board.
_Avoid_: Scoring-board view

**Field mask**:
A geometric region identifying board pixels expected to share one scoring-field
color, excluding expected wire margins.
_Avoid_: Wireframe segmentation

**Dart-likelihood map**:
A per-pixel estimate of how likely a location is to belong to a dart, before
entry-point localization.
_Avoid_: Dart detection result

**Score estimate**:
A best-guess sector, scoring ring, and point value, which may carry explicit
uncertainty.
_Avoid_: Confirmed score, score label
