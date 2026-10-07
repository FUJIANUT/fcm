# T1b diagnostic — alpha sensitivity of `pedrycz_gradient_fcm`

10 seeds, 12 clients x 220 samples, L=10, rounds=50, m=2.0. Same per-seed
init protocol as `scripts/T1B_pedrycz_gradient.py`. Values transcribed verbatim
from the diagnostic sweep run on 2026-09-20.

| scenario | alpha | min-dist | mean ACC | worst ACC | rounds |
|---|---:|---:|---:|---:|---:|
| cluster_skew_hard | 0.002 | 0.418 | 0.9092 | 0.7205 | 50 |
| cluster_skew_hard | 0.005 | 0.289 | 0.9181 | 0.7659 | 50 |
| cluster_skew_hard | 0.01 | 0.654 | 0.9389 | 0.7986 | 49 |
| cluster_skew_hard | 0.02 | 0.709 | 0.9435 | 0.8209 | 48 |
| cluster_skew_hard | 0.05 | 124.905 | 0.9109 | 0.8200 | 50 |
| cluster_skew_hard | 0.1 (frozen) | ~2e14 | 0.9208 | 0.8891 | 50 |
| dirichlet_0.03 | 0.002 | 0.241 | 0.9061 | 0.7386 | 50 |
| dirichlet_0.03 | 0.005 | 0.255 | 0.9108 | 0.7195 | 50 |
| dirichlet_0.03 | 0.01 | 0.236 | 0.8994 | 0.6973 | 48 |
| dirichlet_0.03 | 0.02 | 0.543 | 0.9396 | 0.7709 | 50 |
| dirichlet_0.03 | 0.05 | ~2e7 | 0.9504 | 0.7064 | 50 |
| dirichlet_0.03 | 0.1 (frozen) | ~2e29 | 0.9504 | 0.7064 | 50 |
| quantity_skew_extreme | 0.002 | 1.553 | 0.8286 | 0.6138 | 50 |
| quantity_skew_extreme | 0.005 | 1.174 | 0.8094 | 0.6116 | 50 |
| quantity_skew_extreme | 0.01 | 1.212 | 0.8530 | 0.6513 | 47 |
| quantity_skew_extreme | 0.02 | 5.468 | 0.8605 | 0.6405 | 48 |
| quantity_skew_extreme | 0.05 | ~4e33 | 0.7747 | 0.5149 | 48 |
| quantity_skew_extreme | 0.1 (frozen) | ~3e37 | 0.7565 | 0.4702 | 50 |

Reading: under support skew there is no stable-and-accurate alpha. Small alpha
converges but to a COLLAPSED solution (min-dist ~0.2-0.7 vs true 7.5); alpha
>= ~0.05 makes the fixed-point iteration overshoot (effective per-coordinate
gain 2*alpha*sum_ii beta_ii*M_is/sigma2_iij exceeds 1) and prototypes diverge
to infinity. IID data stays stable at all alpha tested (0.005-0.2) because
every client supports every cluster and sigma2 is well-conditioned.
