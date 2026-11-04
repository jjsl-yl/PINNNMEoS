# Configuration Audit

Authority: the complete manuscript supplied by the author on 2026-11-04.
The original working directory and its historical checkpoints were not edited.

| Item | Release default | Historical implementation |
| --- | --- | --- |
| Parameter scale | Absolute original Table I value | Absolute optimized value, floored at 1 |
| Total field loss | Sum of four batch means | Mean over samples and four fields |
| Proposal sigma | 0.40 x absolute original Table I value | Parameter-specific widths x 2, 3, or 4 |
| Nuclear Esym interval | 27.7-34.1 MeV | Same |
| Astro selection | J0740 + J0437 + GW170817 | Final audit included all three |
| Historical 402/296 counts | Not claimed for this configuration | Obtained with historical proposal rule |

No retrained checkpoint is included. Old checkpoint values cannot be interpreted
using new scales without changing predictions. Portable exports always retain
the scale recorded by the actual checkpoint.

## Settings Not Fully Specified in the Manuscript

Original parameter scales use the displayed (rounded) original Table I numbers.
The proposal mean uses the available full-precision optimized parameters. The
phrase "reference value" in the 40% proposal rule is interpreted as the original
Table I reference, consistently with the normalization paragraph.

The manuscript does not supply Gaussian truncation bounds. This release retains
the bounds used by the existing widened scan, defined as twice each old distance
from the optimized mean. They are executable constants in `wide_r1r2_20261103.py`
and recorded as `hard_bounds` in every manifest. They are **inherited numerical
choices**, not values explicitly stated in the manuscript. They need author
confirmation in a final reproducibility description.

AdamW weight decay 1e-6, gradient clipping at 10, seed 20260619, residual floor
1e6 MeV^3, density input divisor 0.16 fm^-3, saturation search [0.08,0.24],
and nuclear-derivative steps are inherited numerical choices exposed by the
scripts. A saturation minimum outside the search bracket is a numerical
rejection, not a proof that no physical saturation point exists.

The text's Esym=30.9+/-3.2 choice is implemented as requested. This package does
not resolve its attribution to Lattimer & Lim; verify the original literature
before asserting that it is that paper's reported interval.

The manuscript's validation table and training/resource measurements are not
rerun here. Changing scaling and multiplying the total loss by four can change
the AdamW trajectory, clipping, final accuracy and timings. The same applies to
changed proposal distributions and accepted counts. Do not fill new output
files with historical numbers.

## Code Provenance

Training and benchmark entry points originate from the project's local source
scripts. `vendor/rmf` is the local rmf-toolkit snapshot; `tov_love.py` and
`gw_tidal_kde.py` originate from the local nuclear_astro_posterior project.
No external package source was downloaded into this repository. The maintainer
must confirm redistribution rights for the local toolkit and BPS table before
public release; a blanket license must not override existing third-party rights.
