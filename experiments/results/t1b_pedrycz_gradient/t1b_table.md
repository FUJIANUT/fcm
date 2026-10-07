alpha=0.1 (frozen on iid sanity); L=10 local steps for pedrycz; 12 clients x 220 samples; 10 seeds; m=2.0; true BASE_CENTERS min-dist = 7.5

| scenario | method | min-dist | mean ACC | worst ACC |
|---|---|---:|---:|---:|
| iid | pedrycz_gradient | 7.534 +/- 0.038 | 1.0000 | 1.0000 |
| iid | federated_fcm_L5 | 7.535 +/- 0.038 | 1.0000 | 1.0000 |
| iid | federated_fcm_L10 | 7.535 +/- 0.038 | 1.0000 | 1.0000 |
| iid | centralized_fcm | 7.535 +/- 0.038 | 1.0000 | 1.0000 |
| cluster_skew_hard | pedrycz_gradient | 199400318386654.344 +/- 598200576158175.125 | 0.9208 | 0.8891 |
| cluster_skew_hard | federated_fcm_L5 | 1.332 +/- 0.650 | 0.8556 | 0.6568 |
| cluster_skew_hard | federated_fcm_L10 | 1.755 +/- 0.768 | 0.9334 | 0.7959 |
| cluster_skew_hard | centralized_fcm | 7.509 +/- 0.045 | 1.0000 | 1.0000 |
| cluster_skew_overlap | pedrycz_gradient | 2.569 +/- 0.617 | 0.8321 | 0.6423 |
| cluster_skew_overlap | federated_fcm_L5 | 2.101 +/- 0.399 | 0.7421 | 0.5986 |
| cluster_skew_overlap | federated_fcm_L10 | 1.817 +/- 0.612 | 0.7795 | 0.6473 |
| cluster_skew_overlap | centralized_fcm | 3.909 +/- 0.076 | 0.8635 | 0.8018 |
| dirichlet_0.03 | pedrycz_gradient | 179921844091509461234260901888.000 +/- 539694771165973506326920167424.000 | 0.9504 | 0.7064 |
| dirichlet_0.03 | federated_fcm_L5 | 0.664 +/- 0.290 | 0.9134 | 0.6918 |
| dirichlet_0.03 | federated_fcm_L10 | 0.589 +/- 0.297 | 0.9171 | 0.7091 |
| dirichlet_0.03 | centralized_fcm | 6.862 +/- 1.935 | 0.9840 | 0.9509 |
| dirichlet_0.1 | pedrycz_gradient | 17607854422501946.000 +/- 52823249640038200.000 | 0.8491 | 0.5464 |
| dirichlet_0.1 | federated_fcm_L5 | 1.568 +/- 0.695 | 0.9190 | 0.7873 |
| dirichlet_0.1 | federated_fcm_L10 | 1.612 +/- 0.718 | 0.9246 | 0.7732 |
| dirichlet_0.1 | centralized_fcm | 7.487 +/- 0.024 | 1.0000 | 1.0000 |
| overlap_noise | pedrycz_gradient | 3.198 +/- 0.453 | 0.8075 | 0.6964 |
| overlap_noise | federated_fcm_L5 | 3.855 +/- 0.524 | 0.8514 | 0.7809 |
| overlap_noise | federated_fcm_L10 | 3.529 +/- 0.400 | 0.8342 | 0.7450 |
| overlap_noise | centralized_fcm | 4.680 +/- 0.160 | 0.9011 | 0.8691 |
| quantity_skew_extreme | pedrycz_gradient | 30134404915333953833095055913716503543808.000 +/- 82433284515453890174598576568000483164160.000 | 0.7565 | 0.4702 |
| quantity_skew_extreme | federated_fcm_L5 | 1.989 +/- 1.864 | 0.8735 | 0.6949 |
| quantity_skew_extreme | federated_fcm_L10 | 2.217 +/- 1.934 | 0.8756 | 0.6903 |
| quantity_skew_extreme | centralized_fcm | 4.976 +/- 3.099 | 0.9361 | 0.8483 |
