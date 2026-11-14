# Paper ledger

Updated 2026-11-13 19:14:59. Spec: [ONFM 구현 명세서 v1.9](https://claude.ai/artifact/2NQNHTrpjoxtJyBvDmh26D).

| Experiment | Number | Value | Source |
| --- | --- | --- | --- |
| E09 | ODC val-B AUROC, P0 ImageNet ViT-B | 0.924 | `results/e09/e09_eval_summary.json` → auroc.vitb_P0_imagenet.ODC |
| E09 | ODC val-B AUROC, P2 continued MAE | 0.893 | `results/e09/e09_eval_summary.json` → auroc.vitb_P2_cont.ODC |
| E09 | ODC val-B AUROC, P1 MAE from scratch | 0.870 | `results/e09/e09_eval_summary.json` → auroc.vitb_P1_scratch.ODC |
| E09 | ODC delta P2 - P0 [95% CI] | -0.031 [-0.063, -0.002] | `results/e09/e09_eval_summary.json` → delta_vs_P0.vitb_P2_cont.ODC |
| E09 | ODC delta P1 - P0 [95% CI] | -0.054 [-0.099, -0.018] | `results/e09/e09_eval_summary.json` → delta_vs_P0.vitb_P1_scratch.ODC |
| E09 | ON_any delta P1 - P0 [95% CI] | -0.056 [-0.121, +0.003] | `results/e09/e09_eval_summary.json` → delta_vs_P0.vitb_P1_scratch.ON_any |
| E09 | pool v3 size | 104323 | `results/e09/pool_v3_lock.json` → pool_size |
| E11 | test positives (ON_any) | 45 | `results/e11/locked_test_results.json` → positives.ON_any |
| E11 | ON_any test AUROC, A0-384 | 0.912 | `results/e11/locked_test_results.json` → auroc.a0_384.ON_any |
| E11 | ON_any test AUROC, residual ACR-384 | 0.917 | `results/e11/locked_test_results.json` → auroc.racr_384.ON_any |
| E11 | ON_any delta ACR - A0 [95% CI] (primary) | +0.005 [-0.006, +0.019] | `results/e11/locked_test_results.json` → delta['racr_384 - a0_384'].ON_any |
| E11 | ON_any delta ACR - shuffled frames [95% CI] | -0.001 [-0.012, +0.010] | `results/e11/locked_test_results.json` → delta['racr_384 - racr_384_shuffle'].ON_any |
| E11 | primary criterion met | false | `results/e11/locked_test_results.json` → primary_success |
| E10 | config A (v1.3) on 56 dev images | {"median_centre_err_rd": 0.151, "n_gt_0p2": 24, "n": 56} | `data/annotation/fe_disc_v1/e10_eval.json` → A (computed: median e, count e>0.2) |
| E10 | config C (v1.2+v1.3 ensemble + TTA) on 56 dev images | {"median_centre_err_rd": 0.109, "n_gt_0p2": 13, "n": 56} | `data/annotation/fe_disc_v1/e10_eval.json` → C (computed: median e, count e>0.2) |
| FE v1.3 | RIM-ONE external hospitals disc Dice | 0.954 | `results/fe/fe_v13_seed42/metrics.json` → test.rimone.dice_disc |
| FE v1.3 | PAPILA test disc Dice | 0.955 | `results/fe/fe_v13_seed42/metrics.json` → test.papila.dice_disc |
| E08b | result record (seed-wise ODC deltas, internals) | [0.002, 0.005, 0.018] | `results/e08/e08_record_and_prereg.json` → E08b_result.odc_racr_minus_a0_per_seed |
| FE v1.4 | IDRiD test (103) decoded centre error median (r_d) | 0.080 | `results/fe/fe_v14_seed42/metrics.json` → test.idrid_loc.decode_err_median |
| FE v1.4 | IDRiD test (103) images > 0.2 r_d | 1 | `results/fe/fe_v14_seed42/metrics.json` → test.idrid_loc.decode_gt_0p2 |
| FE v1.4 | IDRiD segmentation test disc Dice | 0.969 | `results/fe/fe_v14_seed42/metrics.json` → test.idrid_seg.dice_disc |
| E12 | RFMiD 2.0 kept images | 832 | `results/e12/rfmid2_lock.json` → n_kept |
| E12 | RFMiD 2.0 ON_any2 positives | 43 | `results/e12/rfmid2_lock.json` → positives_kept.ON_any2 |
| E13b | ON_any2 delta disc-zoom - A0 [95% CI] (primary) | +0.011 [-0.029, +0.050] | `results/e13/e13b_locked_results.json` → delta['dz_384 - a0_384'].ON_any2 |
| E13b | ON_any2 AUROC, A0 on RFMiD 2.0 | 0.769 | `results/e13/e13b_locked_results.json` → auroc.a0_384.ON_any2 |
| E13b | primary criterion met | false | `results/e13/e13b_locked_results.json` → primary_success |
| E14 | R0 ON_any on unseen camera B | 0.848 | `results/e14/e14_decision.json` → R0_unseen_camera.foldB_ON_any |
| E14 | R3 - R0 Disease_Risk on camera C | -0.089 | `results/e14/e14_decision.json` → deltas_vs_R0.foldC.R3.Disease_Risk |
| E15 | v1.4p IDRiD test decoded centre error median (r_d) | 0.077 | `results/fe/fe_v14p_seed42/metrics.json` → test.idrid_loc.decode_err_median |
| E15 | v1.4p RIM-ONE test images > 0.2 r_d (of 174) | 87 | `results/fe/fe_v14p_seed42/metrics.json` → test.rimone.decode_gt_0p2 |
| E15b | v1.4p RIM-ONE failure detection AUROC, combined signal | 0.747 | `results/fe/e15b_failure_detection.json` → ['v14p/rimone'].combo.auroc |
| E15b | v1.4p RIM-ONE failure detection AUROC, q_frame | 0.705 | `results/fe/e15b_failure_detection.json` → ['v14p/rimone'].inv_q.auroc |
| E16 | RFMiD 2.0 512x512 subset: ON positives / within-source AUROC | {"n": 511, "on_pos": 43, "auroc": 0.718, "fe_valid": 0.93} | `results/e13/e16_analysis.json` → by_resolution['512x512'] |
| E17 | A0 ON_any original vs matched blur | {"original": 0.923, "512": 0.922, "512_jpeg75": 0.921, "matched": 0.92} | `results/e17/e17_decision.json` → A0_ON_any |
| E17 | resolution augmentation effect on ON_any (matched) [95% CI] | [-0.027, -0.049, -0.006] | `results/e17/e17_decision.json` → RES_minus_A0_ON_any.matched |
| E18 | ON_any delta ACR true - mirrored frame [95% CI] | [-0.002, -0.015, 0.011] | `results/e18/e18_decision.json` → acr_minus_mirror.ON_any |
| E18 | ON_any delta ACR - A0 [95% CI] | [-0.004, -0.018, 0.009] | `results/e18/e18_decision.json` → acr_minus_a0.ON_any |
| E19 | AMD primary: two-stage - additive CME [delta, lo, hi, holm p] | [0.0249, 0.0177, 0.0322, 0.0016] | `results/e19/e19_decision.json` → AMD.primary (from results/e19/AMD/formal_holdout/results.json) |
| E19 | DR primary: two-stage - additive CME [delta, lo, hi, holm p] | [0.0274, 0.0183, 0.0365, 0.0016] | `results/e19/e19_decision.json` → DR.primary (from results/e19/DR/formal_holdout/results.json) |
| E19 | AMD two-stage - OCT only [delta, lo, hi, holm p] | [0.0075, 0.0046, 0.0105, 0.0016] | `results/e19/e19_decision.json` → AMD.2stage_minus_oct |
| E19 | replication criterion met | true | `results/e19/e19_decision.json` → replication_criterion_met |
| E19 | AMD primary: two-stage - additive CME [delta, lo, hi, holm p] | [0.0249, 0.0177, 0.0322, 0.0016] | `results/e19/e19_decision.json` → AMD.primary (from results/e19/AMD/formal_holdout/results.json) |
| E19 | DR primary: two-stage - additive CME [delta, lo, hi, holm p] | [0.0274, 0.0183, 0.0365, 0.0016] | `results/e19/e19_decision.json` → DR.primary (from results/e19/DR/formal_holdout/results.json) |
| E19 | AMD two-stage - OCT only [delta, lo, hi, holm p] | [0.0075, 0.0046, 0.0105, 0.0016] | `results/e19/e19_decision.json` → AMD.2stage_minus_oct |
| E19 | replication criterion met | true | `results/e19/e19_decision.json` → replication_criterion_met |
| E19 review | additive CME - OCT, DR [delta, ci] | -0.026 [-0.039, -0.015] | `results/e19/review_checks.json` → DR.cme_orig_minus_oct |
| E19 review | adapter contribution vs OCT gap, AMD | {"median_abs_cme_minus_s0": 3.3188533782958984, "median_abs_oct_minus_s0": 1.818633794784546, "corr_with_oct_gap": 0.024249618359505044} | `results/e19/review_checks.json` → AMD.adapter_contribution |
| FE review | TTA flag cross-fitted sensitivity, product | 0.708 | `results/fe/review/e21_crossfit.json` → product.sensitivity |
| FE review | RFMiD validity A/B/C, v1.4 seed 44 | {"A": 0.3948692152917505, "B": 0.9376947040498442, "C": 0.33497536945812806} | `results/fe/review/review_steps_3_5.json` → models.fe_v14_seed44.rfmid.valid |
| FE review | IDRiD-RFMiD duplicate pairs | 65 | `results/fe/idrid_rfmid_duplicate_audit.json` → len(pairs_ge_0p985) |

## Files

| Experiment | File | SHA-256 (first 16) |
| --- | --- | --- |
| E05 glaucoma fusion (submitted, Diagnostics) | `results/e05/formal_test/protocol.json` | 5dc99ca5bb8cd89b |
| E05 glaucoma fusion (submitted, Diagnostics) | `results/e05/formal_test/results.json` | 2735e332b12b370d |
| E05 glaucoma fusion (submitted, Diagnostics) | `results/e05/formal_test/supplementary_stats.json` | 23824b29b65ee085 |
| E05 glaucoma fusion (submitted, Diagnostics) | `results/e05/formal_test/fairness_subgroups.csv` | 1cd70be45abb57e1 |
| E06 CME (paper B) | `results/e06/split/holdout_lock.json` | 599bf7d502c2b2b1 |
| E06 CME (paper B) | `results/e06/formal_holdout/protocol.json` | 84854889cb3d2258 |
| E06 CME (paper B) | `results/e06/formal_holdout/results.json` | 06ef0cdb725ecdf5 |
| FE (paper A) | `data/registry/models/onfm-fe-v1.0.0.json` | 120d44ee3ffc3416 |
| FE (paper A) | `data/registry/models/onfm-fe-v1.1.0.json` | 06d534aae06dead0 |
| FE (paper A) | `data/registry/models/onfm-fe-v1.2.0.json` | 12f457dcefd725f0 |
| FE (paper A) | `data/registry/models/onfm-fe-v1.3.0.json` | 9588f43692479978 |
| FE (paper A) | `results/fe/fe_v13_seed42/metrics.json` | 77edf3c8f207ae9f |
| FE (paper A) | `data/annotation/fe_disc_v1/manifest.csv` | c7159b18e24297a3 |
| FE (paper A) | `data/annotation/fe_disc_v1/annotator_record.json` | c3f09507e03d3aae |
| FE (paper A) | `data/annotation/fe_disc_v1/fe_annot_eval.json` | 25c78509df27567a |
| FE (paper A) | `data/annotation/fe_disc_v1/fe_annot_eval_v13.log` | 53ce7881c669f29e |
| FE (paper A) | `fe/fe_decode.py` | 2b5b20b562ee594b |
| FE (paper A) | `fe/fe_infer.py` | c5003716d10faddf |
| E10 FE engineering (TTA, ensemble) | `data/annotation/fe_disc_v1/e10_eval.json` | b9f7dc9f0ba9b60c |
| E10 FE engineering (TTA, ensemble) | `data/annotation/fe_disc_v1/e10_eval.log` | 6db27a3a3bf68e2d |
| E10 FE engineering (TTA, ensemble) | `fe/fe_e10_eval.py` | cb3580cc9fdf4ff5 |
| FE batch-2 locked test (prepared, not run) | `data/annotation/fe_disc_v2/preregistration.json` | 1aa4a509489bfd58 |
| FE batch-2 locked test (prepared, not run) | `data/annotation/fe_disc_v2/manifest.csv` | 0ef3b4d6cf8885c4 |
| FE batch-2 locked test (prepared, not run) | `fe/fe_locked_eval_v2.py` | e9a2199636945e18 |
| Paper 2 first check (RFMiD 1.0 Test exposure) | `results/fe/first_check/step0_label_csvs.txt` | 10c709170e813323 |
| Paper 2 first check (RFMiD 1.0 Test exposure) | `results/fe/first_check/step1_e13_frames.txt` | 6b92c38513aaad94 |
| Paper 2 first check (RFMiD 1.0 Test exposure) | `results/fe/first_check/step2_79pct_source.txt` | de0cc47e28ab3c7e |
| Paper 2 first check (RFMiD 1.0 Test exposure) | `results/fe/first_check/step3_human_exposure.txt` | f9c0ed09f22753b8 |
| Paper 2 first check (RFMiD 1.0 Test exposure) | `results/fe/first_check/step4_e18_and_logs.txt` | 295d43983787b62a |
| Paper 2 camera A/B/C annotation sample (RFMiD 1.0 Test; selection only) | `results/fe/cam_annot/manifest.csv` | 5c077494efa10f1c |
| Paper 2 camera A/B/C annotation sample (RFMiD 1.0 Test; selection only) | `results/fe/cam_annot/selection_log.json` | 095b125764789c8d |
| Paper 2 camera A/B/C annotation sample (RFMiD 1.0 Test; selection only) | `fe/p2_select_annotation_sample.py` | 2630ba578e08e7cf |
| Paper 2 annotation package (blinded, repeats) | `data/annotation/fe_cam_p2/key.csv` | acd697a339f048ff |
| Paper 2 annotation package (blinded, repeats) | `data/annotation/fe_cam_p2/package_log.json` | e6000aa2686bcb2f |
| Paper 2 annotation package (blinded, repeats) | `fe/p2_prepare_annotation.py` | ecd2c08fac7f74f5 |
| FE batch-2 dry-run note (n = 0) | `data/annotation/fe_disc_v2/DRYRUN_NOTE.json` | febe88723851f2a6 |
| Paper 2 locked evaluation (pre-registered, not run) | `results/fe/cam_eval/preregistration.json` | 7ef1f6df958b8c06 |
| Paper 2 locked evaluation (pre-registered, not run) | `fe/p2_cam_locked_eval.py` | 8107871441691549 |
| Paper 2 locked evaluation (pre-registered, not run) | `fe/p2_lock_reference.py` | 76037ba9f6fd175f |
| Paper 2 locked evaluation (pre-registered, not run) | `fe/p2_write_prereg.py` | e6c4048c2215785f |
| Paper 2 locked evaluation (pre-registered, not run) | `fe/p2_eval_selftest.py` | 3ebf1cd7c361c282 |
| Paper 2 locked evaluation (pre-registered, not run) | `results/fe/cam_eval/selftest.log` | f2338296e6046757 |
| Paper 2 amendment 1 (RFMiD annotation optional) | `results/fe/cam_eval/preregistration_amendment_1.json` | d35f8a3638a7a12a |
| Paper 2 Chaksu reference (locked, no FE) | `results/fe/chaksu_ref/reference_lock.json` | 012e379a489396e4 |
| Paper 2 Chaksu reference (locked, no FE) | `results/fe/chaksu_ref/reference_train.csv` | 73fc75f7506eeedf |
| Paper 2 Chaksu reference (locked, no FE) | `results/fe/chaksu_ref/reference_scale_train.csv` | af9944cba762eae7 |
| Paper 2 Chaksu reference (locked, no FE) | `results/fe/chaksu_ref/summary.json` | 642ac677a28aa4f7 |
| Paper 2 Chaksu reference (locked, no FE) | `results/fe/chaksu_ref/scale_summary.json` | 6b82c1ede7a38c12 |
| Paper 2 Chaksu reference (locked, no FE) | `data/eval/chaksu/provenance.json` | 4d774661eddc6e25 |
| Paper 2 Chaksu reference (locked, no FE) | `fe/p2c_build_reference.py` | df1dbe82ff32b812 |
| Paper 2 Chaksu reference (locked, no FE) | `fe/p2c_ref_scale.py` | 5bd7e94a5a41bf97 |
| Paper 2 Chaksu locked evaluation | `results/fe/chaksu_eval/preregistration.json` | a00d1de319ab82d0 |
| Paper 2 Chaksu locked evaluation | `fe/p2c_chaksu_locked_eval.py` | 291a4d7a66b5269c |
| Paper 2 Chaksu locked evaluation | `fe/p2c_write_prereg.py` | ddef78a6ace07cb0 |
| Paper 2 Chaksu locked evaluation | `fe/p2c_selftest.py` | e16c660c9952b65d |
| Paper 2 Chaksu locked evaluation | `results/fe/chaksu_eval/selftest.log` | 0a57ecef3b9f432c |
| Paper 2 Chaksu post hoc development (Train) | `results/fe/chaksu_posthoc/rootcause_summary.json` | c936e3a60d7c9630 |
| Paper 2 Chaksu post hoc development (Train) | `results/fe/chaksu_posthoc/dev_fundus_crop.json` | f83eaa784efc7063 |
| Paper 2 Chaksu post hoc development (Train) | `results/fe/chaksu_posthoc/dev_seed_committee.json` | ab0831ee3e7b4666 |
| Paper 2 Chaksu post hoc development (Train) | `results/fe/chaksu_posthoc/dev_tta_selection.json` | 361b609ab01ef0ec |
| Paper 2 Chaksu post hoc development (Train) | `results/fe/chaksu_eval/secondary_rank_agreement.json` | 8595ba98d5d76962 |
| Paper 2 Chaksu post hoc development (Train) | `fe/p2c_posthoc_rootcause.py` | 2054e3d414a4a24b |
| Paper 2 Chaksu post hoc development (Train) | `fe/p2c_dev_fundus_crop.py` | dc918af243a6c545 |
| Paper 2 Chaksu post hoc development (Train) | `fe/p2c_dev_seed_committee.py` | 33625042c6d9d103 |
| Paper 2 Chaksu post hoc development (Train) | `fe/p2c_dev_tta_selection.py` | 7d08eb9f00bb6123 |
| Paper 2 Chaksu post hoc development (Train) | `fe/p2c_rank_agreement.py` | e61402fc4c551dd8 |
| Paper 2 Chaksu Test reference (locked, no FE) | `results/fe/chaksu_ref_test/reference_lock.json` | e6d8a14577d4baf6 |
| Paper 2 Chaksu Test reference (locked, no FE) | `results/fe/chaksu_ref_test/reference_test.csv` | 5cbec4370755b51d |
| Paper 2 Chaksu Test reference (locked, no FE) | `results/fe/chaksu_ref_test/reference_scale_test.csv` | b77b3f3666de37aa |
| Paper 2 Chaksu Test reference (locked, no FE) | `fe/p2c_build_reference_test.py` | 0afef3f885096ed9 |
| Paper 2 Chaksu Test reference (locked, no FE) | `fe/p2c_ref_scale_test.py` | 47c3e38e9c1b25e3 |
| Paper 2 Chaksu Test confirmation | `results/fe/chaksu_test_eval/preregistration.json` | a7d1fa6b709508fe |
| Paper 2 Chaksu Test confirmation | `fe/p2c_test_locked_eval.py` | d7453a8bfe1a41c0 |
| Paper 2 Chaksu Test confirmation | `fe/p2c_crop.py` | f3c8bcbd92f6d08b |
| Paper 2 Chaksu Test confirmation | `fe/p2c_test_selftest.py` | 78e45ddd47920f39 |
| Paper 2 Chaksu Test confirmation | `fe/p2c_test_write_prereg.py` | 2850c1e58a51dc69 |
| Paper 2 Chaksu Test confirmation | `results/fe/chaksu_test_eval/selftest.log` | 123422b8b6f8ac46 |
| Paper 2 RFMiD supplementary (amendment 2) | `results/fe/cam_eval/preregistration_amendment_2.json` | f327c38f613d1712 |
| Paper 2 RFMiD supplementary (amendment 2) | `results/fe/rfmid_supp/labelfree_selection_record.json` | b40ccffd553a85a2 |
| Paper 2 RFMiD supplementary (amendment 2) | `fe/p2r_labelfree_selection.py` | f328a034ad879b09 |
| Paper 2 RFMiD supplementary (amendment 2) | `fe/p2r_make_calib_package.py` | a3132e437654378c |
| Paper 2 RFMiD supplementary (amendment 2) | `data/annotation/fe_cam_p2_calib/key.csv` | 44df0495114dcb1d |
| Paper 2 RFMiD supplementary (amendment 2) | `data/annotation/fe_cam_p2_calib/package_log.json` | d186790178ce339e |
| Paper 2 exploratory E3/E4 | `results/fe/chaksu_explore/E3_E4_selection.json` | 4c459dd965fe51f0 |
| Paper 2 exploratory E3/E4 | `fe/p2c_explore_selection.py` | 6519c5d86e1f55a7 |
| Paper 2 amendment 3 (no human annotation) | `results/fe/cam_eval/preregistration_amendment_3.json` | 3db3ee66874c1ed4 |
| Paper 2 E2 RIGA reference (locked, no FE) | `results/fe/riga_ref/reference_lock.json` | 6371013a2d2a5276 |
| Paper 2 E2 RIGA reference (locked, no FE) | `results/fe/riga_ref/reference_riga.csv` | e560093c43b78610 |
| Paper 2 E2 RIGA reference (locked, no FE) | `results/fe/riga_ref/summary.json` | fb9bdde56c4b45ef |
| Paper 2 E2 RIGA reference (locked, no FE) | `fe/p2g_build_reference.py` | cda07c793fa30e4c |
| Paper 2 E2 RIGA reference (locked, no FE) | `fe/p2g_reference_summary.py` | adcabce9289d3d5f |
| Paper 2 E2 RIGA reference (locked, no FE) | `data/eval/riga/SHA256SUMS.txt` | 2bf2a7504e236224 |
| Paper 2 E2 RIGA evaluation | `results/fe/riga_eval/preregistration.json` | 68514e772ea611e9 |
| Paper 2 E2 RIGA evaluation | `fe/p2g_riga_locked_eval.py` | eca1f87f5a1ace9a |
| Paper 2 E2 RIGA evaluation | `fe/p2g_selftest.py` | 1ee8f34b3eea61b5 |
| Paper 2 E2 RIGA evaluation | `fe/p2g_write_prereg.py` | 3d3b14a3ece4dc04 |
| Paper 2 E2 RIGA evaluation | `results/fe/riga_eval/selftest.log` | d2b8bbb15a349ce1 |
| Paper 2 E6/E6b exploratory (consistent-but-wrong, full gate) | `results/fe/chaksu_explore/E6_consistent_wrong.json` | 08672c9c0ee4353c |
| Paper 2 E6/E6b exploratory (consistent-but-wrong, full gate) | `results/fe/chaksu_explore/E6b_full_gate.json` | 36caf6d5b8ee4032 |
| Paper 2 E6/E6b exploratory (consistent-but-wrong, full gate) | `fe/p2e_consistent_wrong.py` | d855096f1462e30e |
| Paper 2 E8 PALM reference (locked, no FE) | `results/fe/palm_ref/reference_lock.json` | 01ad849356688179 |
| Paper 2 E8 PALM reference (locked, no FE) | `results/fe/palm_ref/reference_palm.csv` | 3b0a2896943ff7ab |
| Paper 2 E8 PALM reference (locked, no FE) | `results/fe/palm_ref/exclusions_clusters.json` | 2c55355a88937114 |
| Paper 2 E8 PALM reference (locked, no FE) | `results/fe/palm_ref/dup_audit.json` | f00dea3a381a199c |
| Paper 2 E8 PALM reference (locked, no FE) | `results/fe/palm_ref/summary.json` | 4c632a40878b0b55 |
| Paper 2 E8 PALM reference (locked, no FE) | `data/eval/palm/SHA256SUMS.txt` | c9bd096c1678c7e5 |
| Paper 2 E8 PALM evaluation | `results/fe/palm_eval/preregistration.json` | 0c7fe00c55c67c45 |
| Paper 2 E8 PALM evaluation | `fe/p2p_palm_eval.py` | f5b5eb103077cd3b |
| Paper 2 E8 PALM evaluation | `fe/p2p_selftest.py` | 22a9eeb088a1a272 |
| Paper 2 E8 PALM evaluation | `fe/p2p_write_prereg.py` | 28d31bf484c557dd |
| Paper 2 E8 PALM evaluation | `results/fe/palm_eval/selftest.log` | b8a67e36136ef06c |
| E07 pretraining pilot | `results/e07/pool_lock.json` | 5092b67f04b9fcc1 |
| E07 pretraining pilot | `results/e07/pool_lock_v1_SUPERSEDED.json` | 2df32cc76c0848b7 |
| E07 pretraining pilot | `results/e07/pool_manifest.csv` | 5a9d603d9af69619 |
| E07 pretraining pilot | `results/e07/pilot_conclusion.json` | 3e892ad27d2eaf10 |
| E07 pretraining pilot | `results/e07/finetune_val_split.json` | 449a62d8ae7245e8 |
| E07 pretraining pilot | `results/e07/finetune_results.json` | 1cd47ed6417fcfe1 |
| E07 pretraining pilot | `results/e07/e07_decision.json` | cdcbbb16bff42e56 |
| E07 pretraining pilot | `results/e07/probe_val_only.log` | 146ddea706124433 |
| E08 ACR | `results/e08/e08_record_and_prereg.json` | 5cd6cc23f1146201 |
| E08 ACR | `results/e08/acr_results.json` | 2f0e7c1ee5dcba4a |
| E08 ACR | `results/e08/e08b_results.json` | cd6d958932c0e3b5 |
| E08 ACR | `e08/e08b_racr.py` | 977f57327772bc57 |
| E09 large-scale pretraining | `results/e09/pool_v3_lock.json` | 7ae6456deb94c220 |
| E09 large-scale pretraining | `results/e09/pool_v3_manifest.csv` | 1a6b29cfee9e94ee |
| E09 large-scale pretraining | `results/e09/e09_preregistration.json` | 6c9a48a9e8d692fd |
| E09 large-scale pretraining | `results/e09/finetune_preds.json` | c8fed1094430e51f |
| E09 large-scale pretraining | `results/e09/e09_eval_summary.json` | 669596edb66cc32c |
| E09 large-scale pretraining | `results/e09/e09_decision.json` | e81ba285cae81f0c |
| E09 large-scale pretraining | `results/e09/mae_vitb16_imagenet_ep100/history.json` | d6a36c1ccb1f1350 |
| E09 large-scale pretraining | `results/e09/mae_vitb16_scratch_ep300/history.json` | 07fd3dd6bf501ed2 |
| E11 design-v1 locked test | `results/e11/preregistration.json` | 3d9e834f5faf936e |
| E11 design-v1 locked test | `results/e11/locked_test_results.json` | dca66d0bdfb5997b |
| E11 design-v1 locked test | `results/e11/e11_decision.json` | d3cb8b96b0dafacf |
| E11 design-v1 locked test | `results/e11/e11_locked_test.log` | 4773964df86862b8 |
| E11 design-v1 locked test | `e11/e11_locked_test.py` | b9e292f21aa811e1 |
| FE v1.4 (IDRiD; camera generalisation) | `data/registry/models/onfm-fe-v1.4.0.json` | 7f9a192ab1564323 |
| FE v1.4 (IDRiD; camera generalisation) | `results/fe/fe_v14_seed42/metrics.json` | ab7ccdea954fdf6d |
| FE v1.4 (IDRiD; camera generalisation) | `results/fe/fe_prepare_v14.log` | 8cbaf41913e47e44 |
| FE v1.4 (IDRiD; camera generalisation) | `results/e13/rfmid_frames_v14.json` | b2e1d22c0ff818ab |
| FE v1.4 (IDRiD; camera generalisation) | `results/e13/frame_v14.log` | 2a034681fba30d15 |
| FE v1.4 (IDRiD; camera generalisation) | `results/e13/rfmid_frames_by_config.json` | 17e46116e90dbafc |
| E12 RFMiD 2.0 external lock | `results/e12/rfmid2_lock.json` | 802b818aec6bb559 |
| E12 RFMiD 2.0 external lock | `results/e12/rfmid2_manifest.csv` | 9271a7686274828b |
| E12 RFMiD 2.0 external lock | `results/e12/rfmid2_frames_v14.json` | 2b72fe73bc01d37f |
| E12 RFMiD 2.0 external lock | `e12/e12_build_rfmid2_lock.py` | 943e5adf970a4b6f |
| E13 disc-zoom second view | `results/e13/e13a_record.json` | ca30212801bbb967 |
| E13 disc-zoom second view | `results/e13/e13a_preds.json` | df9e00de5440c5ab |
| E13 disc-zoom second view | `results/e13/e13b_preregistration.json` | 5cbd6c06665c5763 |
| E13 disc-zoom second view | `results/e13/e13b_locked_results.json` | 0732bd0bb2c2e2eb |
| E13 disc-zoom second view | `results/e13/e13b_decision.json` | f3eb1004a1d16dcd |
| E13 disc-zoom second view | `e13/e13a_dev.py` | fafe44e69f8971cb |
| E13 disc-zoom second view | `e13/e13b_locked.py` | a44e040da4f530d0 |
| E13 disc-zoom second view | `e13/e13_disc_cache.py` | 7a48f2515fa4369a |
| E14 leave-one-camera-out | `results/e14/e14_preds.json` | 695aa63d15b90110 |
| E14 leave-one-camera-out | `results/e14/e14_decision.json` | 0c5cfb884637038c |
| E14 leave-one-camera-out | `results/e14/e14.log` | 0dfefacbd1076897 |
| E14 leave-one-camera-out | `e14/e14_loco.py` | 1fefec7540d34324 |
| E14 leave-one-camera-out | `e14/e14_prepare.py` | 921caf91a2022d8d |
| E15 product-track FE (v1.4p) | `data/registry/models/onfm-fe-v1.4p.0.json` | d7023a136740c804 |
| E15 product-track FE (v1.4p) | `results/fe/fe_v14p_seed42/metrics.json` | fe424344c2d65347 |
| E15 product-track FE (v1.4p) | `results/e13/rfmid_frames_v14p.json` | 6d76482d7770d2dc |
| E15 product-track FE (v1.4p) | `results/e13/frame_v14p.log` | 6148d67587d0eae8 |
| E15b FE failure self-detection | `results/fe/e15b_failure_detection.json` | 533d00dd83708a1d |
| E15b FE failure self-detection | `results/fe/e15b_decision.json` | 6801aa5641113ed9 |
| E15b FE failure self-detection | `fe/fe_e15b_failure.py` | a58b3950d91f3dce |
| E16 RFMiD 2.0 drop analysis | `results/e13/e16_analysis.json` | 457c7a6056fc43a4 |
| E16 RFMiD 2.0 drop analysis | `results/e13/e16_analysis.log` | 920d8a35df120842 |
| E16 RFMiD 2.0 drop analysis | `e13/e16_rfmid2_analysis.py` | 15d51bbde9ba0670 |
| E17 image-quality robustness | `results/e17/e17_preds.json` | c6268475558e5070 |
| E17 image-quality robustness | `results/e17/e17_decision.json` | cf5946292b5d5227 |
| E17 image-quality robustness | `results/e17/matched_condition.json` | 61ce7db2864e1fbc |
| E17 image-quality robustness | `results/e17/prepare.log` | e968ae9644aebef2 |
| E17 image-quality robustness | `e17/e17_res.py` | bcf204ccf99c4f2a |
| E17 image-quality robustness | `e17/e17_prepare.py` | d956fbf715ebaea8 |
| E17 image-quality robustness | `e17/e17_matched.py` | 4ab6bc548c2658a2 |
| E18 ACR retest (FE v1.4 frames, mirrored control) | `results/e18/e18_plan.json` | f524dc137b616e69 |
| E18 ACR retest (FE v1.4 frames, mirrored control) | `results/e18/e18_preds.json` | 2296ea7b71895e2b |
| E18 ACR retest (FE v1.4 frames, mirrored control) | `results/e18/e18_decision.json` | 6e42537dc253384b |
| E18 ACR retest (FE v1.4 frames, mirrored control) | `results/e18/rfmid_test_frames_v14.json` | 29ab0e1bb85f66fe |
| E18 ACR retest (FE v1.4 frames, mirrored control) | `e18/e18_acr_retest.py` | 86edef1700abec96 |
| E19 CME replication (FairVision AMD, DR) | `results/e19/e19_test_lock.json` | cd98ce10dc51cb6e |
| E19 CME replication (FairVision AMD, DR) | `results/e19/e19_preregistration.json` | 28abb10187d08b3b |
| E19 CME replication (FairVision AMD, DR) | `results/e19/e19_decision.json` | 3c49557efbd80259 |
| E19 CME replication (FairVision AMD, DR) | `results/e19/AMD/formal_holdout/protocol.json` | 05ccdd799d82fb45 |
| E19 CME replication (FairVision AMD, DR) | `results/e19/AMD/formal_holdout/results.json` | c3cd774a96f95886 |
| E19 CME replication (FairVision AMD, DR) | `results/e19/AMD/split/holdout_lock.json` | ee93e6905c097b62 |
| E19 CME replication (FairVision AMD, DR) | `results/e19/AMD/cache/training_meta.csv` | de101879f85d5bb8 |
| E19 CME replication (FairVision AMD, DR) | `results/e19/DR/formal_holdout/protocol.json` | 49605d6ff1f100bd |
| E19 CME replication (FairVision AMD, DR) | `results/e19/DR/formal_holdout/results.json` | 4e27f90e43767d58 |
| E19 CME replication (FairVision AMD, DR) | `results/e19/DR/split/holdout_lock.json` | 03d068fa3f7379ed |
| E19 CME replication (FairVision AMD, DR) | `results/e19/DR/cache/training_meta.csv` | 883fae70eabfff56 |
| E19 CME replication (FairVision AMD, DR) | `e19/e19_prepare.py` | f90a79546d5da2ba |
| E19 CME replication (FairVision AMD, DR) | `e19/run_e19.sh` | f9ad2f7b1640444e |
| E19 CME replication (FairVision AMD, DR) | `e06/e06_cme.py` | 36c3ebf82cf6418a |
| E19 CME replication (FairVision AMD, DR) | `e06/e06_cme2.py` | 9b494ac1a2251b59 |
| E19 CME replication (FairVision AMD, DR) | `e06/e06_formal_holdout.py` | 5d3f3a393e953bc7 |
| B2 inference pipeline v1 | `data/registry/models/onfm-pipeline-v1.0.0.json` | 8358cd8da81c2b45 |
| B2 inference pipeline v1 | `models/onfm-fundus-a0-v1/manifest.json` | 6cb8f6f073a690f3 |
| B2 inference pipeline v1 | `models/onfm-fundus-a0-v1/calibration.json` | 6f167a43328e860b |
| B2 inference pipeline v1 | `onfm/flag_thresholds.json` | 87b701072d441007 |
| B2 inference pipeline v1 | `onfm/MODEL_CARD.md` | a92223af15dacaa8 |
| B2 inference pipeline v1 | `onfm/run.py` | 6adc818c3a31a791 |
| B2 inference pipeline v1 | `onfm/frame.py` | 4893da26151eea29 |
| B2 inference pipeline v1 | `onfm/fundus.py` | 5fd5106f182d486c |
| B2 inference pipeline v1 | `onfm/preprocess.py` | e71ddc36d17bc565 |
| B2 inference pipeline v1 | `onfm/cme.py` | ef3b3a63459a1468 |
| B2 inference pipeline v1 | `onfm/report.py` | ab35ab21ae142c90 |
| B2 inference pipeline v1 | `results/onfm_calibrate_flags.log` | 6484e263e531e906 |
| B2 inference pipeline v1 | `results/e20_release.log` | 5430378c2372d3de |
| FE paper review (duplicates, radius basis, seeds, fractions, cross-fit, ensemble) | `results/fe/idrid_rfmid_duplicate_audit.json` | 8f11964ace7c2418 |
| FE paper review (duplicates, radius basis, seeds, fractions, cross-fit, ensemble) | `results/fe/idrid_internal_leak.json` | 7ca2626d6a48538d |
| FE paper review (duplicates, radius basis, seeds, fractions, cross-fit, ensemble) | `results/fe/idrid_test_clean.json` | 330205170eee7111 |
| FE paper review (duplicates, radius basis, seeds, fractions, cross-fit, ensemble) | `results/fe/radius_basis.json` | a9d677a99adcbe81 |
| FE paper review (duplicates, radius basis, seeds, fractions, cross-fit, ensemble) | `results/fe/review/e21_crossfit.json` | 4fdee8cbbc933595 |
| FE paper review (duplicates, radius basis, seeds, fractions, cross-fit, ensemble) | `results/fe/review/summary_all_models.json` | 21e2439790345017 |
| FE paper review (duplicates, radius basis, seeds, fractions, cross-fit, ensemble) | `results/fe/review/seed_ensemble.json` | 4cffa3b3a564dc67 |
| FE paper review (duplicates, radius basis, seeds, fractions, cross-fit, ensemble) | `results/fe/review/review_steps_3_5.json` | bfe2f1974337d213 |
| FE paper review (duplicates, radius basis, seeds, fractions, cross-fit, ensemble) | `fe/fe_audit_idrid_rfmid.py` | 461228142ffacf62 |
| FE paper review (duplicates, radius basis, seeds, fractions, cross-fit, ensemble) | `fe/fe_audit_idrid_internal.py` | b26284b9799b8992 |
| FE paper review (duplicates, radius basis, seeds, fractions, cross-fit, ensemble) | `fe/fe_radius_basis.py` | a3264956101bf557 |
| FE paper review (duplicates, radius basis, seeds, fractions, cross-fit, ensemble) | `fe/fe_e21_crossfit.py` | 4317a32b2c527d11 |
| FE paper review (duplicates, radius basis, seeds, fractions, cross-fit, ensemble) | `fe/fe_review_summary.py` | d30160d3549ba0ba |
| FE paper review (duplicates, radius basis, seeds, fractions, cross-fit, ensemble) | `fe/fe_seed_ensemble.py` | a4f6c1146f86f069 |
| FE paper review (duplicates, radius basis, seeds, fractions, cross-fit, ensemble) | `fe/run_fe_review.sh` | a8b4634ab5f1db48 |
| FE paper review (duplicates, radius basis, seeds, fractions, cross-fit, ensemble) | `tools/progress.py` | 526ff49632bd59a8 |
| Review and audits 2026-11-09/10 (user-run) | `results/e09/finetune_summary_ep30.json` | aab9ab45fe63eef7 |
| Review and audits 2026-11-09/10 (user-run) | `results/e09/finetune_summary_ep50.json` | 8d6890060a975ba3 |
| Review and audits 2026-11-09/10 (user-run) | `results/e13/e22_label_check.json` | 0acdf806c4d2e14d |
| Review and audits 2026-11-09/10 (user-run) | `results/e19/negative_flips.json` | 480f4ab0cf1196a3 |
| Review and audits 2026-11-09/10 (user-run) | `results/e19/paper3_results_all.json` | 2863caa1d9c2d685 |
| Review and audits 2026-11-09/10 (user-run) | `results/e19/learning_curves.csv` | 3b7aa496b2bca766 |
| Review and audits 2026-11-09/10 (user-run) | `results/e19/architecture_source.txt` | 02cb57e7897d663e |
| Paper 3 review 2 (validation thresholds) | `results/e19/negative_flips_valthr.json` | fc243874498032df |
| Paper 3 review 2 (validation thresholds) | `results/e19/negative_flips_valthr.log` | 0909df927d6b4ff8 |
| Paper 3 review 2 (validation thresholds) | `e19/e19_flips_valthr.py` | 711c6e25214f78c4 |
| Paper 3 review 3 (supplementary tables) | `results/e19/paper3_tableS2_pvalues.csv` | 430b586a7cce69ca |
| Paper 3 review 3 (supplementary tables) | `results/e19/paper3_tableS4_timeline.csv` | b9fbd559ff923e9d |
| Paper 3 review 3 (supplementary tables) | `results/e19/paper3_calibration.json` | 6ff853cc7c5ac0ff |
| E23 post hoc sensitivity (adapter, PCT) | `results/e23/e23a_summary.json` | fcecf294e22c0dcb |
| E23 post hoc sensitivity (adapter, PCT) | `results/e23/e23b_summary.json` | 9d0fe191594909b0 |
| E23 post hoc sensitivity (adapter, PCT) | `e23/run_e23a.sh` | 52961315e8e10117 |
| E23 post hoc sensitivity (adapter, PCT) | `e23/run_e23b.sh` | dda3f697e2182a13 |
| E23 post hoc sensitivity (adapter, PCT) | `e23/e23a_summary.py` | 575353bd6564dc5c |
| E23 post hoc sensitivity (adapter, PCT) | `e23/e23b_pct.py` | b304a70b10e51cb7 |
| E23 post hoc sensitivity (adapter, PCT) | `e23/e23b_eval.py` | e97199f09e4f1074 |
| E23 post hoc sensitivity (adapter, PCT) | `results/e23_a.log` | 5bb8c4fb2a3ecc85 |
| E23 post hoc sensitivity (adapter, PCT) | `results/e23_b.log` | 1a437bb12905e31f |
| Paper 3 final (E23, Figure 3) | `results/e19/paper3_Figure3.png` | add39b873392be3f |
| Paper 3 reviews 4-5 and paper 1 supplement check | `results/e19/paper3_review4_checks.json` | 224a4409d176a3e9 |
| Paper 3 reviews 4-5 and paper 1 supplement check | `results/e19/paper3_compute_cost.json` | e2352b2f813f9f38 |
| Paper 3 reviews 4-5 and paper 1 supplement check | `results/e19/paper3_review5.json` | 7bc093cabf965e32 |
| Paper 3 reviews 4-5 and paper 1 supplement check | `results/e23/e23b_summary_1v.json` | d97e2a5a50c43dac |
| Paper 3 reviews 4-5 and paper 1 supplement check | `results/e05/formal_test/confusion_from_predictions.json` | 2ab9a9c413e133da |
| Paper 3 reviews 4-5 and paper 1 supplement check | `e19/e19_paper3_review4.py` | b9ef472036890b4f |
| Paper 3 reviews 4-5 and paper 1 supplement check | `e19/e19_compute_cost.py` | acada09b6ba26351 |
| Paper 3 reviews 4-5 and paper 1 supplement check | `e19/e19_paper3_review5.py` | 2c7d48b9f0eae016 |
| Paper 3 reviews 4-5 and paper 1 supplement check | `e23/run_e23c.sh` | da67b23e60bdd8fc |
| Paper 3 reviews 4-5 and paper 1 supplement check | `results/e23_c.log` | 5baf7b740dcd7c14 |
| E24 post hoc design analyses | `results/e24/e24_designmap.json` | 301f5e665921aaed |
| E24 post hoc design analyses | `results/e24/e24_protocol.json` | cf1276848bdd6d12 |
| E24 post hoc design analyses | `results/e24/e24a_tradeoff.json` | 2ad5472b0db7a33d |
| E24 post hoc design analyses | `results/e24/e24b_decision.json` | f0c460f9574d64b7 |
| E24 post hoc design analyses | `results/e24/e24b_summary.json` | bc96141d0dd31e0d |
| E24 post hoc design analyses | `results/e24/e24c_crosseval.json` | bf18f1507a565e00 |
| E24 post hoc design analyses | `results/e24/e24c_decision.json` | ea1ff7a14af4e804 |
| E24 post hoc design analyses | `results/e24/e24c_summary.json` | ce2b38b0ade45425 |
| E24 post hoc design analyses | `e24/e24_designmap.py` | 4675fd2aaca5f91b |
| E24 post hoc design analyses | `e24/e24a_tradeoff.py` | 2fcd572331c71bf6 |
| E24 post hoc design analyses | `e24/e24b_octonly_adapter.py` | ac30ab642d1634d0 |
| E24 post hoc design analyses | `e24/e24b_summary.py` | c9c6a66df842c02d |
| E24 post hoc design analyses | `e24/e24b_variants.py` | e4b9341d815376a8 |
| E24 post hoc design analyses | `e24/e24c_bscan.py` | 78f0d2a61dfdd5fc |
| E24 post hoc design analyses | `e24/e24c_crosseval.py` | f6564deebc9dcf1e |
| E24 post hoc design analyses | `e24/e24c_roleswap.py` | 6081b2d6e9cfd112 |
| E24 post hoc design analyses | `e24/e24c_summary.py` | 034fa75fd11f621a |
| E24 post hoc design analyses | `e24/run_e24b.sh` | e296256f31a0cf56 |
| E24 post hoc design analyses | `e24/run_e24bc.sh` | 1e2a1baa5ee1f2ad |
| data registry | `data/registry/registry.json` | 3607e6a8f7237aa8 |
