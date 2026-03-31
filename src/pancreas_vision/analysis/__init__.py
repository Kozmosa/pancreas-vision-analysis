"""Analysis subpackage for error analysis and hard-case management."""

from __future__ import annotations

from pancreas_vision.analysis.error_analysis import (
    aggregate_errors_by_source_bucket,
    analyze_hard_case_bags,
    extract_gan_patch_candidates,
    generate_gan_review_shortlist,
    load_attention_summary,
    load_bag_predictions,
    write_error_analysis_outputs,
)

__all__ = [
    "aggregate_errors_by_source_bucket",
    "analyze_hard_case_bags",
    "extract_gan_patch_candidates",
    "generate_gan_review_shortlist",
    "load_attention_summary",
    "load_bag_predictions",
    "write_error_analysis_outputs",
]