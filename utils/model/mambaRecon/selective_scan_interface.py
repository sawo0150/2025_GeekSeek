# mambaRecon/selective_scan_interface.py
"""
Unified selective_scan wrapper
 ├ BACKEND="auto"  : (기본) CUDA → mambapy → reference 순으로 자동
 ├ BACKEND="cuda"  : CUDA 커널 강제 사용, 없으면 RuntimeError
 ├ BACKEND="py"    : mambapy(Pure-PyTorch) 강제
 └ BACKEND="ref"   : state-spaces-mamba 의 reference(Python) 강제
환경변수 `MAMBA_SSM_BACKEND` 로도 덮어쓸 수 있다.
"""

import os, torch
from contextlib import suppress

# ---- 0. 선택 플래그 -------------------------------------------------
BACKEND = os.getenv("MAMBA_SSM_BACKEND", "auto").lower()   # auto / cuda / py / ref
BACKEND = "auto"  # auto / cuda / py / ref
# --------------------------------------------------------------------

_cuda_fn = _ref_fn = _py_fn = None
with suppress(ImportError):
    from mamba_ssm.ops.selective_scan_interface import (
        selective_scan_fn as _cuda_fn,
        selective_scan_ref as _ref_fn,
    )
with suppress(ImportError):
    from mambapy.ops.selective_scan import selective_scan_r as _py_fn


def _raise(msg):                       # 공통 에러 헬퍼
    raise RuntimeError(f"[selective_scan] {msg}")


def _cpu_fallback(*a, **kw):
    if _py_fn is not None:
        return _py_fn(*a, **kw)
    if _ref_fn is not None:
        return _ref_fn(*a, **kw)
    _raise("no CPU implementation found")


def selective_scan_fn(*args, **kwargs):
    if BACKEND == "cuda":
        if _cuda_fn is None:
            _raise("CUDA backend not installed")
        return _cuda_fn(*args, **kwargs)          # 런타임 오류는 그대로 전파
    if BACKEND == "py":
        if _py_fn is None:
            _raise("mambapy backend not installed")
        return _py_fn(*args, **kwargs)
    if BACKEND == "ref":
        if _ref_fn is None:
            _raise("reference backend not installed")
        return _ref_fn(*args, **kwargs)

    # BACKEND == "auto"  ─────────────────────────────────────────────
    if _cuda_fn is not None and torch.cuda.is_available():
        try:
            return _cuda_fn(*args, **kwargs)
        except (RuntimeError, AssertionError):
            pass                       # sm_61 등 커널 실패 → CPU fallback
    return _cpu_fallback(*args, **kwargs)


# 이름 호환
selective_scan_ref = _ref_fn or _cpu_fallback
