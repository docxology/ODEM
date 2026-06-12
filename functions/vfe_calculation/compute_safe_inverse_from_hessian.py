import torch

def compute(
    H: torch.Tensor,
    *,
    min_eigval: float = 1e-9,
    jitter0: float = 1e-12,
    max_tries: int = 8,
    rtol: float = 1e-12,
    atol: float = 1e-15,
    max_abs: float = 1e12,
) -> torch.Tensor | None:
    """
    Stable PD covariance ≈ H^{-1} for a (noisy) Hessian of the NEGATIVE log joint.
    Returns None if we cannot produce a sane covariance.
    """
    # coerce numerics in case caller passed strings
    min_eigval = float(min_eigval); jitter0 = float(jitter0)
    rtol = float(rtol); atol = float(atol); max_abs = float(max_abs)

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        return None

    device, dtype = H.device, H.dtype
    n = H.shape[0]
    I = torch.eye(n, device=device, dtype=dtype)

    # Sanitize: symmetrize, replace NaN/Inf, soft-clip extremes
    Hs = 0.5 * (H + H.T)
    if not torch.isfinite(Hs).all():
        Hs = torch.nan_to_num(Hs, nan=0.0, posinf=max_abs, neginf=-max_abs)
    Hs = torch.clamp(Hs, min=-max_abs, max=max_abs)

    # If diagonal is tiny/nonfinite, bump it
    d = torch.diag(Hs)
    bad = ~torch.isfinite(d) | (d.abs() < min_eigval)
    if bad.any():
        Hs = Hs.clone()
        Hs[bad, bad] = min_eigval

    # 1) Cholesky with escalating jitter (fastest, best)
    jitter = 0.0
    for _ in range(max_tries):
        try:
            Hp = Hs if jitter == 0.0 else Hs + jitter * I
            L = torch.linalg.cholesky(Hp)
            cov = torch.cholesky_solve(I, L)
            cov = 0.5 * (cov + cov.T)
            return cov
        except RuntimeError:
            jitter = jitter0 if jitter == 0.0 else jitter * 10.0

    # Move to CPU for spectral fallbacks (more forgiving)
    H_cpu = Hs.detach().cpu()

    # 2) Eig fallback with relative floor
    try:
        evals, evecs = torch.linalg.eigh(H_cpu)
        lam_floor = max(min_eigval, rtol * float(evals.abs().max().item()) + atol)
        evals = torch.clamp(evals, min=lam_floor)
        inv = 1.0 / evals
        cov_cpu = (evecs * inv) @ evecs.T
        cov = cov_cpu.to(device=device, dtype=dtype)
        cov = 0.5 * (cov + cov.T)
        if torch.isfinite(cov).all():
            return cov
    except RuntimeError:
        cov = None

    # 3) SVD last resort
    H_cpu = torch.nan_to_num(H_cpu, nan=0.0, posinf=max_abs, neginf=-max_abs)
    H_cpu = torch.clamp(H_cpu, min=-max_abs, max=max_abs)
    if not torch.isfinite(H_cpu).all():
        return None

    U, S, Vh = torch.linalg.svd(H_cpu, full_matrices=False)
    s_floor = max(min_eigval, rtol * float(S.max().item()) + atol)
    S = torch.clamp(S, min=s_floor)
    invS = 1.0 / S
    cov_cpu = (Vh.T * invS) @ Vh
    cov = cov_cpu.to(device=device, dtype=dtype)
    cov = 0.5 * (cov + cov.T)
    return cov if torch.isfinite(cov).all() else None
