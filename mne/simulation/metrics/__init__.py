"""Metrics module for compute stc-based metrics."""

from .metrics import (source_estimate_quantification,
                      cosine_score,
                      region_localization_error,
                      precision_score, recall_score,
                      f1_score, roc_auc_score,
                      peak_position_error,
                      spacial_deviation_error,
                      _thresholding, _check_threshold, _uniform_stc)
