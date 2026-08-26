# Balanced-1000 targeted-arm evidence

## Collection audit

The seen-garment source-discovery campaign completed exactly 1,000 valid
attempts, balanced to 250 attempts per garment category. Infrastructure aborts
were excluded from the valid denominator and replenished.

| Category | Valid attempts | Newly accepted successes |
| --- | ---: | ---: |
| Pant long | 250 | 0 |
| Pant short | 250 | 1 |
| Top long | 250 | 1 |
| Top short | 250 | 0 |

The `2/1000` result describes only this new source-discovery campaign. It does
not replace the previously sealed round-2 source of 116 accepted episodes. The
new campaign used the same 12K checkpoint, checkpoint artifact digest, runtime
image, code revision, and CPU cloth device as the older source. The material
runtime difference was the collection path: the new campaign used fresh
`server_cpu` canonical attempts, while the older accepted round-2 episodes came
from `persistent_collection` runs (including `mild_geometry`). Therefore the
two accepted counts are not interchangeable success-rate denominators.

Future broad collection should use an early success-rate circuit breaker. A
campaign whose authenticated valid-attempt sample is far below its reviewed
minimum should stop before exhausting its full attempt budget.

## Success-replay arm

The source contains 116 unique verified successful episodes. The immutable
selection balances distinct H16 windows, not episode counts:

| Category | Unique episodes | Selected H16 windows |
| --- | ---: | ---: |
| Pant long | 12 | 270 |
| Pant short | 54 | 270 |
| Top long | 27 | 270 |
| Top short | 23 | 270 |

The final runtime mixture is 90% organizer BC and 10% success replay, with an
exact batch-64 quota of 58 BC and 6 replay samples. Its immutable mixture ID is
`e1dcb5bc968c68e00c1d140b8ed821d6133c78c4e009b5844b361f73e0e3e83e`.
The final readback-verified revision is
`fc027fdd82adb345931c7717f6a11bfd9b27267c`.

## Hard-state arm

The authenticated recovery audit supplies 33 H16 continuation windows across
the three weak categories:

| Category | Unique recovery episodes | Selected H16 windows |
| --- | ---: | ---: |
| Pant long | 2 | 11 |
| Pant short | 0 | 0 |
| Top long | 3 | 11 |
| Top short | 4 | 11 |

Pant short is absent because the audit had no authenticated pant-short recovery
window and pant short was already the strongest category. The final runtime
mixture is 90% organizer BC and 10% hard-state continuation, with an exact
batch-64 quota of 58 BC and 6 recovery samples. Its immutable mixture ID is
`536fecab8376ed74da425f1904ba2fc63336e346200c75095bcdbe1ece133720`.
The final readback-verified revision is
`6504dea8888ec715358d0a68665cce1b914a35f8`.

## Remaining evidence

Both arms still require independent 2,000-step training followed by the same
fixed seen-80 and unseen-80 evaluation matrices. Checkpoint publication and
evaluation results must be appended only after immutable readback verification.
