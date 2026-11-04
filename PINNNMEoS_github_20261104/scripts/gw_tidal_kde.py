"""Four-dimensional GW mass-tidal KDEs with physical boundary reflection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import KernelDensity


def _reflect(samples: np.ndarray) -> np.ndarray:
    """Reflect q at 1 and both tidal deformabilities at zero."""
    x = np.asarray(samples, dtype=float)
    blocks = []
    for q_flip in (False, True):
        for l1_flip in (False, True):
            for l2_flip in (False, True):
                y = x.copy()
                if q_flip:
                    y[:, 1] = 2.0 - y[:, 1]
                if l1_flip:
                    y[:, 2] *= -1.0
                if l2_flip:
                    y[:, 3] *= -1.0
                blocks.append(y)
    return np.vstack(blocks)


def _reflect_mass(samples: np.ndarray) -> np.ndarray:
    """Reflect the two-dimensional (chirp mass, q) sample at q=1."""
    x = np.asarray(samples, dtype=float)
    reflected = x.copy()
    reflected[:, 1] = 2.0 - reflected[:, 1]
    return np.vstack((x, reflected))


@dataclass
class GWTidalKDE:
    """Weighted KDE in (source chirp mass, q, Lambda1, Lambda2)."""

    mean: np.ndarray
    scale: np.ndarray
    model: KernelDensity

    @classmethod
    def fit(
        cls,
        samples: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        bandwidth: float = 0.10,
        mean: np.ndarray | None = None,
        scale: np.ndarray | None = None,
    ) -> "GWTidalKDE":
        x = np.asarray(samples, dtype=float)
        if x.ndim != 2 or x.shape[1] != 4:
            raise ValueError("samples must have shape (N,4)")
        good = np.all(np.isfinite(x), axis=1) & (x[:, 1] > 0.0) & (x[:, 1] <= 1.0) & (x[:, 2] >= 0.0) & (x[:, 3] >= 0.0)
        x = x[good]
        w = None if weights is None else np.asarray(weights, dtype=float)[good]
        if mean is None:
            mean = np.average(x, axis=0, weights=w)
        if scale is None:
            scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=w))
        scale = np.maximum(np.asarray(scale, dtype=float), 1e-12)
        reflected = _reflect(x)
        z = (reflected - mean) / scale
        reflected_weights = None if w is None else np.tile(w, 8)
        model = KernelDensity(kernel="gaussian", bandwidth=bandwidth, algorithm="ball_tree")
        model.fit(z, sample_weight=reflected_weights)
        return cls(np.asarray(mean), scale, model)

    def log_density(self, points: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(points, dtype=float))
        valid = (p[:, 1] > 0.0) & (p[:, 1] <= 1.0) & (p[:, 2] >= 0.0) & (p[:, 3] >= 0.0)
        out = np.full(len(p), -np.inf)
        if np.any(valid):
            out[valid] = self.model.score_samples((p[valid] - self.mean) / self.scale)
        # The Jacobian is constant for one fitted KDE and can be retained for
        # auditable normalization/comparison with another KDE.
            out[valid] -= np.log(self.scale).sum()
        return out


@dataclass
class GWDensityRatio:
    posterior: GWTidalKDE
    prior: GWTidalKDE

    @classmethod
    def fit(
        cls,
        posterior_samples: np.ndarray,
        prior_samples: np.ndarray,
        *,
        bandwidth: float = 0.12,
    ) -> "GWDensityRatio":
        prior = np.asarray(prior_samples, dtype=float)
        mean = np.mean(prior, axis=0)
        scale = np.std(prior, axis=0)
        return cls(
            GWTidalKDE.fit(posterior_samples, bandwidth=bandwidth, mean=mean, scale=scale),
            GWTidalKDE.fit(prior_samples, bandwidth=bandwidth, mean=mean, scale=scale),
        )

    def log_likelihood(self, points: np.ndarray) -> np.ndarray:
        return self.posterior.log_density(points) - self.prior.log_density(points)


@dataclass
class GWMassPriorKDE:
    """KDE for the source-frame (chirp mass, q) marginal PE prior."""

    mean: np.ndarray
    scale: np.ndarray
    model: KernelDensity

    @classmethod
    def fit(cls, samples: np.ndarray, *, bandwidth: float = 0.28) -> "GWMassPriorKDE":
        x = np.asarray(samples, dtype=float)
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError("mass-prior samples must have shape (N,2)")
        good = np.all(np.isfinite(x), axis=1) & (x[:, 0] > 0.0) & (x[:, 1] > 0.0) & (x[:, 1] <= 1.0)
        x = x[good]
        mean = np.mean(x, axis=0)
        scale = np.maximum(np.std(x, axis=0), 1e-12)
        model = KernelDensity(kernel="gaussian", bandwidth=bandwidth, algorithm="ball_tree")
        model.fit((_reflect_mass(x) - mean) / scale)
        return cls(mean, scale, model)

    def log_density(self, points: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(points, dtype=float))
        valid = np.all(np.isfinite(p), axis=1) & (p[:, 0] > 0.0) & (p[:, 1] > 0.0) & (p[:, 1] <= 1.0)
        out = np.full(len(p), -np.inf)
        if np.any(valid):
            out[valid] = self.model.score_samples((p[valid] - self.mean) / self.scale)
            out[valid] -= np.log(self.scale).sum()
        return out


@dataclass
class GWFactoredPriorLikelihood:
    """GW likelihood with a 4D posterior and factored analytic tidal prior.

    The released GW190425 prior factorizes, to sampling precision, into a
    source-frame mass marginal and uniform component tidal priors.  Estimating
    only the two-dimensional mass marginal avoids taking the ratio of two
    sparse four-dimensional KDEs.  The constant density of the uniform tidal
    prior is irrelevant to relative log likelihoods but its support is kept.
    """

    posterior: GWTidalKDE
    mass_prior: GWMassPriorKDE
    lambda_1_bounds: tuple[float, float]
    lambda_2_bounds: tuple[float, float]

    @classmethod
    def fit(
        cls,
        posterior_samples: np.ndarray,
        mass_prior_samples: np.ndarray,
        *,
        posterior_bandwidth: float = 0.28,
        mass_prior_bandwidth: float = 0.28,
        lambda_1_bounds: tuple[float, float] = (0.0, 5000.0),
        lambda_2_bounds: tuple[float, float] = (0.0, 10000.0),
    ) -> "GWFactoredPriorLikelihood":
        posterior = np.asarray(posterior_samples, dtype=float)
        return cls(
            posterior=GWTidalKDE.fit(posterior, bandwidth=posterior_bandwidth),
            mass_prior=GWMassPriorKDE.fit(mass_prior_samples, bandwidth=mass_prior_bandwidth),
            lambda_1_bounds=lambda_1_bounds,
            lambda_2_bounds=lambda_2_bounds,
        )

    def log_likelihood(self, points: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(points, dtype=float))
        l1_lo, l1_hi = self.lambda_1_bounds
        l2_lo, l2_hi = self.lambda_2_bounds
        inside = (
            np.all(np.isfinite(p), axis=1)
            & (p[:, 2] >= l1_lo)
            & (p[:, 2] <= l1_hi)
            & (p[:, 3] >= l2_lo)
            & (p[:, 3] <= l2_hi)
        )
        out = np.full(len(p), -np.inf)
        if np.any(inside):
            out[inside] = self.posterior.log_density(p[inside]) - self.mass_prior.log_density(p[inside, :2])
        return out
