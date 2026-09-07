"""Evaluate fixed-model responses without paper-specific selection or I/O."""
import numpy as np
from .finite_budget import orthogonal_basis, coefficients, truncate, sparse_response_fit


def response_errors(model, x, targets, basis_names, methods, budgets, *,
                    baseline, evaluation_displacements, calibration_displacements,
                    forward_budget: int, backward_budget: int,
                    steps: int = 32, radius: float = 0.01):
    """Evaluate held-out responses; return rows without choosing a representation.

All tensors share original input coordinates. The caller must supply disjoint
calibration/scoring perturbation banks. Forward visits count examples, not batches.
"""
    import torch
    if x.ndim != 3 or baseline.shape != x.shape or len(targets) != len(x):
        raise ValueError("expected x/baseline [N,C,T] and explicit targets [N]")
    for bank in (evaluation_displacements, calibration_displacements):
        if bank.ndim != 4 or bank.shape[0] != len(x) or bank.shape[2:] != x.shape[1:]:
            raise ValueError("perturbation banks must be [N,Q,C,T] in original coordinates")
        if not torch.isfinite(bank).all():
            raise ValueError("nonfinite perturbation bank")
    if torch.equal(evaluation_displacements, calibration_displacements):
        raise ValueError("calibration and scoring banks must not be reused")
    matrices = {name: orthogonal_basis(x.shape[-1], name) for name in basis_names}
    rows = []
    for n, sample in enumerate(x):
        target = int(targets[n])
        if target < 0:
            raise ValueError("negative target index")
        ve, vc = evaluation_displacements[n], calibration_displacements[n]
        with torch.no_grad():
            original = model(sample[None])
            scored = model(sample[None] - ve)
            if original.ndim != 2 or scored.shape != (len(ve), original.shape[1]) or target >= original.shape[1]:
                raise ValueError("model output or frozen target incompatible")
            truth = (original[0, target] - scored[:, target]).cpu().numpy()
            cal_response = None
            if "surrogate" in methods:
                cal_response = (original[0, target] - model(sample[None] - vc)[:, target]).cpu().numpy()
            if not np.isfinite(truth).all() or (cal_response is not None and not np.isfinite(cal_response).all()):
                raise ValueError("nonfinite intervention response")
        for name, matrix in matrices.items():
            a = torch.as_tensor(matrix, dtype=x.dtype, device=x.device)
            with torch.no_grad():
                # Preservation is checked on both ends of the same intervention.
                points = torch.cat((sample[None], sample[None] - ve), dim=0)
                restored = (points @ a.T) @ a
                if not torch.allclose(restored, points, atol=2e-5, rtol=2e-5):
                    raise ValueError("representation failed reconstruction")
                if not torch.allclose(model(restored), model(points), atol=2e-5, rtol=2e-5):
                    raise ValueError("representation failed fixed-model-output preservation")
            design_eval = (ve @ a.T).flatten(1).cpu().numpy()
            design_cal = (vc @ a.T).flatten(1).cpu().numpy()
            for method in methods:
                counts = ({"forward": len(vc) + 1, "backward": 0} if method == "surrogate"
                          else {"forward": 1 if method == "gradient" else steps if method == "path_gradient" else 2 * sample.numel(),
                                "backward": 1 if method == "gradient" else steps if method == "path_gradient" else 0})
                if counts["forward"] > forward_budget or counts["backward"] > backward_budget:
                    raise ValueError(f"{method} exceeds the declared forward/backward query budget")
                raw = None
                if method != "surrogate":
                    raw, counts = coefficients(model, sample, target, matrix, method=method,
                        baseline=baseline[n], steps=steps, radius=radius)
                for k in budgets:
                    vector = (sparse_response_fit(design_cal, cal_response, k) if method == "surrogate"
                              else truncate(raw, k).ravel())
                    error = float(np.mean((design_eval @ vector - truth) ** 2))
                    if not np.isfinite(error):
                        raise ValueError("nonfinite response error")
                    rows.append(dict(sample=n, representation=name, method=method, k=int(k),
                        mse=error, forward=counts["forward"], backward=counts["backward"],
                        evaluation_forward=1 + len(ve), output_kind="sensitivity_response"))
    return rows
