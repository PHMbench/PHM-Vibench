"""Small numerical primitives for fixed-model, common-intervention explanations.

Coefficients are sensitivities, not normalized heatmaps or IG contributions.
The caller owns data splitting, experiment selection, checkpoints and reporting.
"""
from itertools import combinations
import numpy as np


def orthogonal_basis(length: int, name: str) -> np.ndarray:
    """Return a real square basis; dense storage is intended for bounded windows."""
    if not isinstance(length, int) or not 1 <= length <= 4096:
        raise ValueError("dense basis requires an integer length in [1, 4096]")
    eye = np.eye(length)
    if name == "identity":
        return eye
    if name == "dct":
        n = np.arange(length)
        matrix = np.cos(np.pi * (n[None, :] + 0.5) * n[:, None] / length)
        matrix *= np.sqrt(2.0 / length)
        matrix[0] /= np.sqrt(2.0)
        return matrix
    if name.startswith("haar_packet:"):
        depth = int(name.split(":", 1)[1])
        if depth < 1 or length % (2 ** depth):
            raise ValueError("Haar packet depth must divide the declared window length")
        bands = [eye]
        for _ in range(depth):
            bands = [part for band in bands for part in
                     ((band[::2] + band[1::2]) / np.sqrt(2),
                      (band[::2] - band[1::2]) / np.sqrt(2))]
        return np.concatenate(bands, axis=0)
    raise ValueError(f"unsupported representation: {name}")


def truncate(coefficients: np.ndarray, k: int) -> np.ndarray:
    """Keep k scalar coordinates over all channels, with deterministic ties."""
    coefficients = np.asarray(coefficients, dtype=float)
    if not np.isfinite(coefficients).all() or not 1 <= k <= coefficients.size:
        raise ValueError("finite coefficients and 1 <= k <= number of coordinates required")
    ids = np.argsort(-np.abs(coefficients.ravel()), kind="stable")[:k]
    result = np.zeros(coefficients.size)
    result[ids] = coefficients.ravel()[ids]
    return result.reshape(coefficients.shape)


def exact_sparse_risk(basis: np.ndarray, second_moment: np.ndarray,
                      cross_moment: np.ndarray, response_second_moment: float,
                      k: int) -> tuple[float, np.ndarray]:
    """Exact subset oracle for small problems, not a scalable explainer.

M=E[v v^T], b=E[v d(v)], c=E[d(v)^2]. M is not a centred covariance.
"""
    basis = np.asarray(basis, dtype=float)
    p = basis.shape[0]
    if basis.shape != (p, p) or p > 16 or not 1 <= k <= p:
        raise ValueError("exact oracle requires square p<=16 and 1<=k<=p")
    moment = np.asarray(second_moment, dtype=float)
    cross = np.asarray(cross_moment, dtype=float)
    if moment.shape != (p, p) or cross.shape != (p,):
        raise ValueError("second moment or cross moment has incompatible dimensions")
    if not np.allclose(moment, moment.T) or np.linalg.eigvalsh(moment).min() < -1e-10:
        raise ValueError("second moment must be symmetric positive semidefinite")
    gram, transformed = basis @ moment @ basis.T, basis @ cross
    best, answer = float("inf"), None
    for support in combinations(range(p), k):
        ids = np.array(support)
        block = gram[np.ix_(ids, ids)]
        coeff = np.linalg.pinv(block, hermitian=True) @ transformed[ids]
        if not np.allclose(block @ coeff, transformed[ids], atol=1e-8):
            raise ValueError("cross moment is outside the Gram range; invalid moments")
        risk = float(response_second_moment - transformed[ids] @ coeff)
        if risk < -1e-8:
            raise ValueError("moments imply negative squared error")
        if risk < best:
            best = max(0.0, risk)  # Roundoff only; inconsistent moments fail above.
            answer = np.zeros(p)
            answer[ids] = coeff
    return best, answer


def sparse_response_fit(displacements: np.ndarray, responses: np.ndarray,
                        k: int) -> np.ndarray:
    """Greedy sparse least squares on calibration queries, never scoring queries."""
    design = np.asarray(displacements, dtype=float)
    response = np.asarray(responses, dtype=float)
    if design.ndim != 2 or response.shape != (len(design),):
        raise ValueError("expected calibration design [Q,p] and response [Q]")
    if not np.isfinite(design).all() or not np.isfinite(response).all():
        raise ValueError("calibration values must be finite")
    if not 1 <= k <= min(design.shape):
        raise ValueError("sparse fit requires 1 <= k <= min(Q,p)")
    residual, support = response.copy(), []
    norms = np.linalg.norm(design, axis=0)
    if np.count_nonzero(norms) < k:
        raise ValueError("fewer than k nonzero calibration directions")
    for _ in range(k):
        gain = np.full(design.shape[1], -np.inf)
        eligible = norms > 0
        gain[eligible] = np.abs(design[:, eligible].T @ residual) / norms[eligible]
        gain[support] = -np.inf
        support.append(int(np.argmax(gain)))
        coeff = np.linalg.lstsq(design[:, support], response, rcond=None)[0]
        residual = response - design[:, support] @ coeff
    answer = np.zeros(design.shape[1])
    answer[support] = coeff
    return answer


def coefficients(model, x, target: int, basis, *, method: str,
                 baseline=None, steps: int = 32, radius: float = 0.01):
    """Return [C,T] sensitivity coefficients and counted model-example visits.

x is a single [C,T] tensor. The model must return [batch, classes].
path_gradient is the IG integrand average, not vanilla IG attribution.
secant uses declared finite differences and is not reported as Occlusion.
"""
    import torch
    if x.ndim != 2 or not torch.isfinite(x).all():
        raise ValueError("x must be a finite [C,T] tensor")
    if not isinstance(target, int) or target < 0:
        raise ValueError("an explicit nonnegative target index is required")
    if model.training:
        raise ValueError("model must be frozen in eval mode")
    matrix = torch.as_tensor(basis, dtype=x.dtype, device=x.device)
    if matrix.shape != (x.shape[-1], x.shape[-1]):
        raise ValueError("basis length does not match the declared time coordinates")

    def score(sample):
        output = model(sample.unsqueeze(0))
        if output.ndim != 2 or output.shape[0] != 1 or target >= output.shape[1]:
            raise ValueError("model output must be [1, classes] and contain the fixed target")
        value = output[0, target]
        if not torch.isfinite(value):
            raise ValueError("nonfinite fixed-target model output")
        return value

    if method in {"gradient", "path_gradient"}:
        if method == "gradient":
            points = [x]
        else:
            if baseline is None or baseline.shape != x.shape or steps < 1:
                raise ValueError("path_gradient requires an explicit same-shape baseline and steps>=1")
            # Midpoint quadrature; the baseline and step count define the path method.
            points = [baseline + ((i + 0.5) / steps) * (x - baseline)
                      for i in range(steps)]
        gradients = []
        with torch.enable_grad():
            for point in points:
                query = point.detach().requires_grad_(True)
                gradients.append(torch.autograd.grad(score(query), query)[0].detach())
        raw = torch.stack(gradients).mean(dim=0) @ matrix.T
        visits = {"forward": len(points), "backward": len(points)}
    elif method == "secant":
        if radius <= 0:
            raise ValueError("positive finite-difference radius required")
        raw = torch.empty_like(x)
        with torch.no_grad():
            for channel in range(x.shape[0]):
                for coordinate in range(x.shape[1]):
                    step = torch.zeros_like(x)
                    step[channel] = radius * matrix[coordinate]
                    raw[channel, coordinate] = (score(x + step) - score(x - step)) / (2 * radius)
        visits = {"forward": 2 * x.numel(), "backward": 0}
    else:
        raise ValueError(f"no response-decoder implementation for method {method!r}")
    if not torch.isfinite(raw).all():
        raise ValueError("nonfinite sensitivity coefficients")
    return raw.cpu().numpy(), visits
