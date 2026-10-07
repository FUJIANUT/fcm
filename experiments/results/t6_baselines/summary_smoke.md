# T6 external baselines — 2 seed(s), tag='smoke'

frozen: scffcm eta_l=0.2 eta_g=2.0; fednova server_lr=1.0


## cluster_skew_hard

| method | mean_acc | worst_acc | min_dist |
|---|---|---|---|
| post_none_off | 0.9119 | 0.6705 | 2.7424 |
| pre_none_off | 0.9299 | 0.8227 | 4.7634 |
| post_footprint_on | 1.0000 | 1.0000 | 7.7571 |
| pre_footprint_off | 0.9170 | 0.7705 | 4.7174 |
| stallmann_avg2 | 1.0000 | 1.0000 | 7.4314 |
| fednova | 0.9606 | 0.8364 | 2.5302 |
| scffcm | 1.0000 | 1.0000 | 7.4702 |
| centralized_fcm | 1.0000 | 1.0000 | 7.4702 |

## wine

| method | mean_acc | worst_acc | min_dist |
|---|---|---|---|
| post_none_off | 0.9352 | 0.8056 | 2.2948 |
| pre_none_off | 0.9612 | 0.8611 | 3.1674 |
| post_footprint_on | 0.9335 | 0.8333 | 3.5453 |
| pre_footprint_off | 0.9672 | 0.8640 | 3.5507 |
| stallmann_avg2 | 0.9700 | 0.8333 | 3.1828 |
| fednova | 0.9024 | 0.7500 | 1.3826 |
| scffcm | 0.9675 | 0.8611 | 2.9514 |
| centralized_fcm | 0.9675 | 0.8611 | 2.9396 |