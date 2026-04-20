"""Phase-based autofocus package.

Public API preserves the flat `from core.autofocus import ...` imports that the
rest of the codebase relied on before the split. The four submodules split the
concerns:

- `metrics`          : FocusMetric enum, `_calc_metric`, direction helpers.
- `evaluator`        : Fast evaluator closure + complex-field downsampling.
- `search_classic`   : Linear sweep, Golden Section, Coarse-to-Fine, Robust C2F.
- `search_adaptive`  : Gradient / Ratio / Bracketing / Distance + `AdaptiveFocusState`.
- `analysis`         : Auto-select metric, landscape scan, benchmark.
"""
from .metrics import (
    FocusMetric,
    _calc_metric,
    _get_hf_mask,
    _hf_mask_cache,
    _is_minimize,
    _phase_of,
    _wrap_diff,
)
from .evaluator import (
    AutoFocusResult,
    AutofocusCancelled,
    _make_fast_evaluator,
    downsample_complex_field,
)
from .search_classic import (
    GoldenSearchResult,
    RobustSearchResult,
    autofocus_zscan,
    coarse_to_fine_search,
    golden_section_search,
    robust_coarse_to_fine_search,
)
from .search_adaptive import (
    AdaptiveDistanceResult,
    AdaptiveFocusState,
    AdaptiveStepResult,
    adaptive_bracketing_search,
    adaptive_distance_search,
    adaptive_gradient_search,
    adaptive_ratio_search,
)
from .analysis import (
    AutofocusBenchmarkResult,
    BenchmarkEntry,
    MetricLandscapeResult,
    auto_select_metric,
    autofocus_benchmark,
    scan_metric_landscape,
)

__all__ = [
    # metrics
    "FocusMetric",
    "_calc_metric",
    "_get_hf_mask",
    "_hf_mask_cache",
    "_is_minimize",
    "_phase_of",
    "_wrap_diff",
    # evaluator
    "AutoFocusResult",
    "AutofocusCancelled",
    "_make_fast_evaluator",
    "downsample_complex_field",
    # classic search
    "GoldenSearchResult",
    "RobustSearchResult",
    "autofocus_zscan",
    "coarse_to_fine_search",
    "golden_section_search",
    "robust_coarse_to_fine_search",
    # adaptive search
    "AdaptiveDistanceResult",
    "AdaptiveFocusState",
    "AdaptiveStepResult",
    "adaptive_bracketing_search",
    "adaptive_distance_search",
    "adaptive_gradient_search",
    "adaptive_ratio_search",
    # analysis
    "AutofocusBenchmarkResult",
    "BenchmarkEntry",
    "MetricLandscapeResult",
    "auto_select_metric",
    "autofocus_benchmark",
    "scan_metric_landscape",
]
