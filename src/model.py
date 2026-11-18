"""Damped-wave modal basis and least-squares fitting."""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse.csgraph import connected_components


def graph_modes(laplacian):
    """An orthonormal graph basis with only the global constant removed.

    Retained zero modes are explicit contrasts between connected components.
    Positive modes retain their Laplacian eigenvectors. In particular, the
    connected case returns exactly the same eigendecomposition slice as before.
    """
    operator = np.asarray(laplacian, dtype=float)
    if (operator.ndim != 2 or operator.shape[0] != operator.shape[1]
            or not np.isfinite(operator).all()
            or not np.allclose(operator, operator.T, rtol=0, atol=1e-10)):
        raise ValueError("Require a finite symmetric graph Laplacian")
    weights = -operator.copy()
    np.fill_diagonal(weights, 0)
    count, membership = connected_components(weights > 0, directed=False)
    eigenvalues, eigenvectors = np.linalg.eigh(operator)
    tolerance = 100 * np.finfo(float).eps * max(1., np.linalg.norm(operator))
    if (np.max(np.abs(eigenvalues[:count])) > tolerance
            or np.any(eigenvalues[count:] <= 0)
            or np.max(np.abs(operator.sum(axis=1))) > tolerance):
        raise ValueError("Laplacian nullspace disagrees with its components")
    if count == 1:
        return eigenvalues[1:], eigenvectors[:, 1:]
    contrasts = np.zeros((len(operator), count - 1))
    # Weighted Helmert contrasts of component indicators. Their entries follow
    # actual graph membership, including after an arbitrary parcel permutation.
    for k in range(1, count):
        previous = membership < k
        current = membership == k
        a, b = int(previous.sum()), int(current.sum())
        contrasts[previous, k - 1] = np.sqrt(b / (a * (a + b)))
        contrasts[current, k - 1] = -np.sqrt(a / (b * (a + b)))
    return (np.r_[np.zeros(count - 1), eigenvalues[count:]],
            np.column_stack((contrasts, eigenvectors[:, count:])))


def coefficient_modes(eigenvalues):
    """Map temporal-design columns to modes: one static, two per positive mode."""
    values = np.asarray(eigenvalues)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Require finite nonnegative retained eigenvalues")
    return np.repeat(np.arange(len(values)), np.where(values == 0, 1, 2))


def temporal_basis(time, eigenvalues, tau, normalize=True):
    """Vectorized conserved modal design, with the parcel-fit critical limit."""
    eigenvalues = np.asarray(eigenvalues)
    coefficient_modes(eigenvalues)  # Validate the mode specification.
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("Require a finite positive relaxation time")
    time = np.asarray(time)[:, None]
    d = 1 - 4 * tau * eigenvalues
    zero = eigenvalues == 0
    critical = (np.abs(d) <= 1e-6) & ~zero
    under = (d < 0) & ~critical & ~zero
    over = (d >= 0) & ~critical & ~zero
    functions = np.empty((len(time), len(eigenvalues), 2))
    functions[:, zero, 0] = 1
    functions[:, zero, 1] = 0
    env = np.exp(-time / (2 * tau))
    omega = np.sqrt(-d[under]) / (2 * tau)
    functions[:, under, 0] = env * np.cos(time * omega)
    functions[:, under, 1] = env * np.sin(time * omega)
    roots = np.sqrt(d[over])
    functions[:, over, 0] = np.exp(time * ((-1 + roots) / (2 * tau)))
    functions[:, over, 1] = np.exp(time * ((-1 - roots) / (2 * tau)))
    functions[:, critical, 0] = env
    functions[:, critical, 1] = time * env
    keep = np.repeat(~zero, 2)
    keep[::2] = True
    functions = functions.reshape(len(time), -1)[:, keep]
    if normalize:
        functions /= np.linalg.norm(functions, axis=0)
    return functions


def basis(time, lam, tau, scaling=False):
    if lam == 0:
        return np.ones((len(time), 1))
    disc = 1 - 4 * tau * lam
    if scaling and abs(disc) <= 1e-6:
        env = np.exp(-time / (2 * tau))
        return np.column_stack((env, time * env))
    if disc < 0:
        omega = np.sqrt(-disc) / (2 * tau)
        env = np.exp(-time / (2 * tau))
        return np.column_stack((env * np.cos(omega * time),
                                env * np.sin(omega * time)))
    r1 = (-1 + np.sqrt(disc)) / (2 * tau)
    r2 = (-1 - np.sqrt(disc)) / (2 * tau)
    return np.column_stack((np.exp(r1 * time), np.exp(r2 * time)))


def fit(signal, time, laplacian, resolution, scaling=False):
    """Modal least squares with static component contrasts and one shared tau."""
    eigenvalues, eigenvectors = graph_modes(laplacian)
    z, *_ = np.linalg.lstsq(resolution @ eigenvectors, signal, rcond=None)
    mean_var = np.mean(np.var(z, axis=1))
    threshold = mean_var * 1e-12 if mean_var > 0 else np.finfo(float).eps
    active = np.std(z, axis=1) > np.sqrt(threshold)

    def residuals(tau_array):
        tau = float(tau_array[0])
        errors = []
        for lam, y, use in zip(eigenvalues, z, active):
            if not use:
                continue
            matrix = basis(time, lam, tau, scaling)
            coefficient, *_ = np.linalg.lstsq(matrix, y, rcond=None)
            errors.append((matrix @ coefficient - y) / np.std(y))
        return np.concatenate(errors) if errors else np.zeros(1)

    initial = 0.05 if scaling else 0.12
    bounds = (0.001, 1.0) if scaling else (0.01, 1.0)
    opt = least_squares(residuals, x0=[initial], bounds=bounds)
    tau = float(opt.x[0])
    if scaling:
        return np.nan, tau
    r2 = []
    for lam, y in zip(eigenvalues, z):
        matrix = basis(time, lam, tau)
        coefficient, *_ = np.linalg.lstsq(matrix, y, rcond=None)
        error = np.sum((y - matrix @ coefficient) ** 2)
        variance = np.sum((y - y.mean()) ** 2)
        r2.append(0.0 if variance < threshold else 1 - error / variance)
    return float(np.mean(r2)), tau
