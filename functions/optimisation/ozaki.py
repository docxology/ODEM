import sys

import torch
from functions import compute_Jacobian
from torch.autograd.functional import jacobian
from functions.vfe_calculation import compute_log_det

import torch

def logabsdet_svd(J: torch.Tensor, floor_ratio: float = 1e-12) -> torch.Tensor:
    """
    Robust log|det(J)| via singular values.
    J: (..., n, n) real tensor.
    floor_ratio: clamp smallest σ to floor_ratio * σ_max to avoid -inf / huge gradients.
    Returns: (...,) tensor (batched OK).
    """
    s = torch.linalg.svdvals(J)  # (..., n), σ_i ≥ 0
    # Build a scale-aware floor to stabilize near-singular cases
    tiny = torch.finfo(J.dtype).tiny
    smax = s.max(dim=-1, keepdim=True).values.clamp_min(torch.tensor(tiny, dtype=J.dtype, device=J.device))
    floor = floor_ratio * smax
    s = s.clamp_min(floor)  # keeps shape/gradients stable
    return torch.sum(torch.log(s), dim=-1)

def compute_t(nu, n, J):
    """
    t = exp(nu) / (|det J|)^(1/n) with robust |det J|.
    """
    logabsdet = logabsdet_svd(J, floor_ratio=1e-12)  # robust and smooth
    alpha_log = logabsdet / n
    t = torch.exp(nu - alpha_log)
    # Final bounds keep the Ozaki step sane
    return t.clamp(1e-6, 1.0)

def _solve_stable(A, B, lam0=1e-9, tries=6):
    I = torch.eye(A.size(-1), device=A.device, dtype=A.dtype)
    lam = lam0
    for _ in range(tries):
        try:
            return torch.linalg.solve(A + lam*I, B)
        except RuntimeError:
            lam *= 10.0
    # Last resort: least-squares
    return torch.linalg.pinv(A) @ B

def integrate(vfe, q, kappa, nu, initial_jitter, device, D, kx=None, dx=None):
    # Drift: h(q) = D q - kappa ∇_q VFE
    h_mu = (D @ q.reshape(kx * dx)).reshape((kx, dx)) - kappa * \
           torch.autograd.grad(outputs=vfe, inputs=q, create_graph=True)[0]

    # Jacobian J = ∂h/∂q, shape (kx*dx, kx*dx)
    J = compute_Jacobian.compute(matrix=h_mu, variable=q)

    # Robust step size
    t = compute_t(nu, kx*dx, J).to(device)

    # Δq = J^{-1}(e^{J t} - I) h  -> do as a solve
    E = torch.matrix_exp(J * t)
    I = torch.eye(J.size(-1), dtype=J.dtype, device=J.device)
    rhs = (E - I) @ h_mu.reshape(kx * dx)
    delta_q_vec = _solve_stable(J, rhs)
    delta_q = delta_q_vec.reshape((kx, dx))

    # Advance
    # if not torch.isfinite(delta_q).all():
    #     print("Tensor contains NaN or Inf values!")
    q = (q + delta_q).detach().requires_grad_(True)
    return q, t, initial_jitter
