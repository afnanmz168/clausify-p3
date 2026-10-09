| configuration | micro-F1 [95% CI] | P | R | F2 | macro-F1 (>=10 pos) | High-risk recall (missed) |
|---|---|---|---|---|---|---|
| original: tfidf_408_C1_t0.5 | 0.775 | 0.742 | 0.810 | 0.796 | 0.666 | 0.648 (62) |
| original: transformer256_408_max_t0.5 | 0.692 | 0.557 | 0.913 | 0.810 | 0.6114 | 0.841 (28) |
| original: and_ensemble_original | 0.779 | 0.774 | 0.784 | 0.782 | 0.6584 | 0.631 (65) |
| tfidf_untuned | 0.773 [0.755, 0.790] | 0.743 | 0.805 | 0.792 | 0.6494 | 0.653 (61) |
| transformer_untuned | 0.717 [0.699, 0.734] | 0.575 | 0.953 | 0.842 | 0.6514 | 0.886 (20) |
| tfidf_tuned_f1 | 0.785 [0.768, 0.801] | 0.750 | 0.824 | 0.808 | 0.6731 | 0.676 (57) |
| transformer_tuned_f1 | 0.797 [0.781, 0.812] | 0.741 | 0.864 | 0.836 | 0.7108 | 0.676 (57) |
| ensemble_tuned_f1 | 0.809 [0.793, 0.825] | 0.864 | 0.761 | 0.780 | 0.6661 | 0.540 (81) |
| tfidf_tuned_f2 | 0.743 [0.727, 0.759] | 0.631 | 0.904 | 0.832 | 0.6535 | 0.795 (36) |
| transformer_tuned_f2 | 0.757 [0.740, 0.773] | 0.638 | 0.929 | 0.852 | 0.6891 | 0.903 (17) |
| ensemble_tuned_f2 | 0.783 [0.768, 0.798] | 0.681 | 0.922 | 0.861 | 0.7025 | 0.869 (23) |

- **tfidf_untuned**: chosen on validation {'C': 1.0, 'threshold': 0.5}
- **transformer_untuned**: chosen on validation {'pool': 'max', 'threshold': 0.5}; vs baseline {'diff': -0.0555, 'ci95': [-0.075187, -0.036425], 'p_gt0': 0.0}
- **tfidf_tuned_f1**: chosen on validation {'C': 10.0, 'global_threshold': 0.5}
- **transformer_tuned_f1**: chosen on validation {'pool': 'max', 'global_threshold': 0.96}; vs baseline {'metric': 'micro-F1', 'diff': 0.0121, 'ci95': [-0.005327, 0.029885], 'p_gt0': 0.912}
- **ensemble_tuned_f1**: chosen on validation {'rule': 'AND', 'validation_scores': {'AND': 0.8574, 'OR': 0.8043, 'MIXED': 0.8441, 'AVG': 0.8495}, 'tfidf_C': 10.0, 'transformer_pool': 'max'}; vs baseline {'metric': 'micro-F1', 'diff': 0.024, 'ci95': [0.011266, 0.036907], 'p_gt0': 1.0}
- **tfidf_tuned_f2**: chosen on validation {'C': 10.0, 'global_threshold': 0.24}
- **transformer_tuned_f2**: chosen on validation {'pool': 'max', 'global_threshold': 0.88}; vs baseline {'metric': 'micro-F2', 'diff': 0.0196, 'ci95': [0.003611, 0.035767], 'p_gt0': 0.992}
- **ensemble_tuned_f2**: chosen on validation {'rule': 'AVG', 'validation_scores': {'AND': 0.8995, 'OR': 0.866, 'MIXED': 0.891, 'AVG': 0.8997}, 'tfidf_C': 10.0, 'transformer_pool': 'max'}; vs baseline {'metric': 'micro-F2', 'diff': 0.0288, 'ci95': [0.017912, 0.039942], 'p_gt0': 1.0}
