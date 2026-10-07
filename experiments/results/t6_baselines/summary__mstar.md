# T6 external baselines — 10 seed(s), tag='_mstar'

frozen: scffcm eta_l=0.2 eta_g=2.0; fednova server_lr=1.0


## cluster_skew_hard

| method | mean_acc | worst_acc | min_dist |
|---|---|---|---|
| post_none_off | 0.9059 | 0.7123 | 2.0713 |
| pre_none_off | 0.9860 | 0.9645 | 7.6100 |
| post_footprint_on | 1.0000 | 1.0000 | 7.6893 |
| pre_footprint_off | 0.9834 | 0.9541 | 7.6726 |
| stallmann_avg2 | 1.0000 | 1.0000 | 7.4275 |
| fednova | 0.9543 | 0.8200 | 2.0750 |
| scffcm | 1.0000 | 1.0000 | 7.4902 |
| centralized_fcm | 1.0000 | 1.0000 | 7.4902 |

## cluster_skew_overlap

| method | mean_acc | worst_acc | min_dist |
|---|---|---|---|
| post_none_off | 0.7304 | 0.5891 | 2.0503 |
| pre_none_off | 0.8599 | 0.8023 | 4.5938 |
| post_footprint_on | 0.8078 | 0.7205 | 4.6593 |
| pre_footprint_off | 0.8481 | 0.7745 | 4.9852 |
| stallmann_avg2 | 0.8628 | 0.7927 | 3.5041 |
| fednova | 0.7258 | 0.5523 | 1.9301 |
| scffcm | 0.8617 | 0.8036 | 3.8352 |
| centralized_fcm | 0.8643 | 0.8123 | 3.8462 |

## digits_pca16

| method | mean_acc | worst_acc | min_dist |
|---|---|---|---|
| post_none_off | 0.4084 | 0.3124 | 1.6185 |
| pre_none_off | 0.6036 | 0.4181 | 2.1586 |
| post_footprint_on | 0.6018 | 0.4167 | 2.3289 |
| pre_footprint_off | 0.6598 | 0.4421 | 2.7043 |
| stallmann_avg2 | 0.5223 | 0.5003 | 6.3758 |
| fednova | 0.3775 | 0.3013 | 1.4475 |
| scffcm | 0.7173 | 0.4862 | 1.2134 |
| centralized_fcm | 0.7510 | 0.4261 | 1.8537 |

## digits_pca32

| method | mean_acc | worst_acc | min_dist |
|---|---|---|---|
| post_none_off | 0.3619 | 0.2754 | 2.1015 |
| pre_none_off | 0.5510 | 0.3736 | 2.4774 |
| post_footprint_on | 0.5649 | 0.4037 | 2.7386 |
| pre_footprint_off | 0.6812 | 0.4822 | 3.2498 |
| stallmann_avg2 | 0.5118 | 0.4992 | 9.7726 |
| fednova | 0.3626 | 0.2869 | 1.9266 |
| scffcm | 0.7086 | 0.5097 | 2.3053 |
| centralized_fcm | 0.7487 | 0.4872 | 2.3744 |

## dirichlet_0.03

| method | mean_acc | worst_acc | min_dist |
|---|---|---|---|
| post_none_off | 0.8956 | 0.6759 | 0.6773 |
| pre_none_off | 0.9871 | 0.9514 | 7.3466 |
| post_footprint_on | 0.9664 | 0.9041 | 5.8704 |
| pre_footprint_off | 0.9580 | 0.9023 | 6.7373 |
| stallmann_avg2 | 1.0000 | 1.0000 | 7.4109 |
| fednova | 0.9117 | 0.7600 | 0.2329 |
| scffcm | 1.0000 | 1.0000 | 7.3275 |
| centralized_fcm | 1.0000 | 1.0000 | 7.5010 |

## dirichlet_0.1

| method | mean_acc | worst_acc | min_dist |
|---|---|---|---|
| post_none_off | 0.8933 | 0.6973 | 1.3528 |
| pre_none_off | 0.9487 | 0.9032 | 6.7447 |
| post_footprint_on | 0.9502 | 0.8945 | 5.7944 |
| pre_footprint_off | 0.9357 | 0.8541 | 6.1267 |
| stallmann_avg2 | 1.0000 | 1.0000 | 7.3670 |
| fednova | 0.9047 | 0.7791 | 1.2027 |
| scffcm | 0.9938 | 0.9586 | 7.2014 |
| centralized_fcm | 0.9841 | 0.9673 | 6.8840 |

## letter

| method | mean_acc | worst_acc | min_dist |
|---|---|---|---|
| post_none_off | 0.2565 | 0.2087 | 1.5141 |
| pre_none_off | 0.2797 | 0.2160 | 1.7782 |
| post_footprint_on | 0.2734 | 0.2301 | 1.7130 |
| pre_footprint_off | 0.2955 | 0.2316 | 1.9735 |
| stallmann_avg2 | 0.3224 | 0.2420 | 1.6319 |
| fednova | 0.2494 | 0.2127 | 1.4626 |
| scffcm | 0.2885 | 0.2095 | 0.9188 |
| centralized_fcm | 0.3045 | 0.1949 | 1.0903 |

## mnist784_pca32 (20c)

| method | mean_acc | worst_acc | min_dist |
|---|---|---|---|
| post_none_off | 0.4030 | 0.3246 | 2.0734 |
| pre_none_off | 0.4717 | 0.4040 | 2.3511 |
| post_footprint_on | 0.4950 | 0.4002 | 2.1012 |
| pre_footprint_off | 0.5469 | 0.4644 | 2.7187 |
| stallmann_avg2 | 0.3535 | 0.3388 | 10.0917 |
| fednova | 0.3932 | 0.3294 | 1.9955 |
| scffcm | 0.5024 | 0.4041 | 2.1106 |
| centralized_fcm | 0.5107 | 0.4347 | 2.2858 |

## mnist784_pca32 (50c)

| method | mean_acc | worst_acc | min_dist |
|---|---|---|---|
| post_none_off | 0.3964 | 0.3261 | 2.1054 |
| pre_none_off | 0.4728 | 0.4020 | 2.3414 |
| post_footprint_on | 0.4828 | 0.3722 | 2.1129 |
| pre_footprint_off | 0.5483 | 0.4550 | 2.6414 |
| stallmann_avg2 | 0.3543 | 0.3377 | 9.7510 |
| fednova | 0.4115 | 0.3374 | 2.0164 |
| scffcm | 0.5199 | 0.4123 | 2.2022 |
| centralized_fcm | 0.5341 | 0.4549 | 2.2720 |

## overlap_noise

| method | mean_acc | worst_acc | min_dist |
|---|---|---|---|
| post_none_off | 0.8713 | 0.8227 | 4.0018 |
| pre_none_off | 0.8818 | 0.8432 | 4.7510 |
| post_footprint_on | 0.8784 | 0.8164 | 4.7728 |
| pre_footprint_off | 0.8850 | 0.8441 | 4.9564 |
| stallmann_avg2 | 0.9014 | 0.8705 | 4.3720 |
| fednova | 0.8309 | 0.7695 | 3.6737 |
| scffcm | 0.9038 | 0.8786 | 4.7972 |
| centralized_fcm | 0.9038 | 0.8786 | 4.7974 |

## pendigits

| method | mean_acc | worst_acc | min_dist |
|---|---|---|---|
| post_none_off | 0.5278 | 0.4570 | 1.5239 |
| pre_none_off | 0.6797 | 0.5108 | 1.8688 |
| post_footprint_on | 0.6373 | 0.4992 | 1.7093 |
| pre_footprint_off | 0.6512 | 0.5045 | 1.7895 |
| stallmann_avg2 | 0.7631 | 0.6108 | 2.1943 |
| fednova | 0.5097 | 0.4075 | 1.5505 |
| scffcm | 0.7433 | 0.6126 | 1.5911 |
| centralized_fcm | 0.7390 | 0.5733 | 1.7389 |

## quantity_skew_extreme

| method | mean_acc | worst_acc | min_dist |
|---|---|---|---|
| post_none_off | 0.8686 | 0.6857 | 1.6190 |
| pre_none_off | 0.8928 | 0.7296 | 3.2140 |
| post_footprint_on | 0.8826 | 0.7029 | 2.8172 |
| pre_footprint_off | 0.8980 | 0.7472 | 3.8086 |
| stallmann_avg2 | 1.0000 | 1.0000 | 7.3672 |
| fednova | 0.9688 | 0.9103 | 2.5107 |
| scffcm | 1.0000 | 1.0000 | 7.4379 |
| centralized_fcm | 0.9625 | 0.9333 | 6.2467 |

## satimage

| method | mean_acc | worst_acc | min_dist |
|---|---|---|---|
| post_none_off | 0.7190 | 0.5842 | 2.5794 |
| pre_none_off | 0.6846 | 0.5241 | 3.2630 |
| post_footprint_on | 0.6749 | 0.4732 | 2.6486 |
| pre_footprint_off | 0.6625 | 0.4959 | 2.5634 |
| stallmann_avg2 | 0.7099 | 0.5349 | 3.5441 |
| fednova | 0.6800 | 0.6006 | 1.9046 |
| scffcm | 0.7044 | 0.5776 | 3.0296 |
| centralized_fcm | 0.7166 | 0.5836 | 3.9831 |

## wine

| method | mean_acc | worst_acc | min_dist |
|---|---|---|---|
| post_none_off | 0.9261 | 0.8056 | 2.1485 |
| pre_none_off | 0.9623 | 0.8789 | 3.1447 |
| post_footprint_on | 0.9314 | 0.8296 | 3.5247 |
| pre_footprint_off | 0.9684 | 0.8951 | 3.6006 |
| stallmann_avg2 | 0.9710 | 0.8895 | 3.1743 |
| fednova | 0.8888 | 0.7404 | 1.2083 |
| scffcm | 0.9667 | 0.8844 | 2.9360 |
| centralized_fcm | 0.9673 | 0.8850 | 2.9396 |