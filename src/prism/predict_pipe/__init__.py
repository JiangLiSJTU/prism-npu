"""
prism.predict_pipe — analytical pipe baseline prediction from GEMM specs.

Public API::

    from prism.predict_pipe import (
        ModelSpec,
        KNOWN_MODELS,
        compute_gemm_ops,
        compute_vector_ops,
        estimate_n_kernels,
        predict_pipe_baseline,
        predict_for_model_yaml,
        assign_confidence,
        fit_host_gap,
        fit_kernel_gap,
        leave_one_model_out_cv,
        fit_all_and_save,
    )

Origin: integrates the Windows reviewer's ``.sisyphus/predict_pipe_v0.1.py``
prototype as a proper PRISM module (see Issue #2). The ``.sisyphus/`` copy
remains as a reference; this module is the authoritative implementation.
"""
from .fit import (
    fit_all_and_save,
    fit_host_gap,
    fit_kernel_gap,
    leave_one_model_out_cv,
)
from .model_spec import (
    KNOWN_MODELS,
    ModelSpec,
    compute_gemm_ops,
    compute_vector_ops,
    estimate_n_kernels,
)
from .predict import (
    assign_confidence,
    predict_for_model_yaml,
    predict_pipe_baseline,
)

__all__ = [
    "ModelSpec",
    "KNOWN_MODELS",
    "compute_gemm_ops",
    "compute_vector_ops",
    "estimate_n_kernels",
    "predict_pipe_baseline",
    "predict_for_model_yaml",
    "assign_confidence",
    "fit_host_gap",
    "fit_kernel_gap",
    "leave_one_model_out_cv",
    "fit_all_and_save",
]
