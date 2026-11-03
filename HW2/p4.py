import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
import scipy.sparse as sp


def S_fast(q, X, Y, xib, yib, a, tol=1e-6):
    result = np.zeros_like(X, dtype=float)

    cutoff = a * np.sqrt(-2*np.log(2*np.pi*a**2*tol))

    x = X[0, :]
    y = Y[:, 0]

    for k in range(len(q)):
        # Grid indices near this IB point
        i0 = max(0, np.searchsorted(x, xib[k] - cutoff))
        i1 = min(len(x), np.searchsorted(x, xib[k] + cutoff, side="right"))

        j0 = max(0, np.searchsorted(y, yib[k] - cutoff))
        j1 = min(len(y), np.searchsorted(y, yib[k] + cutoff, side="right"))

        # Only evaluate nearby grid points
        dx = X[j0:j1, i0:i1] - xib[k]
        dy = Y[j0:j1, i0:i1] - yib[k]

        result[j0:j1, i0:i1] += q[k] * delta_a(dx, dy, a)

    return result


def J_fast(u, X, Y, xib, yib, a, h, tol=1e-6):
    result = np.zeros(len(xib))

    cutoff = a * np.sqrt(-2*np.log(2*np.pi*a**2*tol))

    x = X[0, :]
    y = Y[:, 0]

    for k in range(len(xib)):
        i0 = max(0, np.searchsorted(x, xib[k] - cutoff))
        i1 = min(len(x), np.searchsorted(x, xib[k] + cutoff, side="right"))

        j0 = max(0, np.searchsorted(y, yib[k] - cutoff))
        j1 = min(len(y), np.searchsorted(y, yib[k] + cutoff, side="right"))

        dx = X[j0:j1, i0:i1] - xib[k]
        dy = Y[j0:j1, i0:i1] - yib[k]

        weights = delta_a(dx, dy, a)

        result[k] = (
            np.sum(u[j0:j1, i0:i1] * weights) * h**2
        )

    return result

def delta_a(dx, dy, a):
    return np.exp(-(dx**2 + dy**2) / (2*a**2)) / (2*np.pi*a**2)

def S(q, X, Y, xib, yib, a):
    result = np.zeros_like(X)

    for k in range(len(q)):
        dx = X - xib[k]
        dy = Y - yib[k]

        result += q[k] * delta_a(dx, dy, a)

    return result

def J(u, X, Y, xib, yib, a, h):
    result = np.zeros(len(xib))

    for k in range(len(xib)):
        dx = X - xib[k]
        dy = Y - yib[k]

        weights = delta_a(dx, dy, a)

        result[k] = np.sum(u * weights) * h**2

    return result

def IB_finite_diff(f, N, a, b, exact, homogeneous_bndry = False, fast=False):
    """Solve u_xx + u_yy = f with Dirichlet boundary conditions and immersed boundary."""
    if N < 2:
        raise ValueError("N must be at least 2.")

    h = (b - a) / N
    N_ib = round(2*np.pi / h)

    x = np.linspace(a, b, N + 1)
    y = np.linspace(a, b, N + 1)
    theta = np.linspace(a, b, N_ib, endpoint=False)
    x_ib = np.pi + np.cos(theta)
    y_ib = np.pi + np.sin(theta)

    n = N+1

    X, Y = np.meshgrid(x, y)
    a = 1.2 * h

    Dx = sp.diags([1 , -2 , 1], [ -1 , 0 , 1], shape = (n, n), dtype=None) / h**2
    Dy = sp.diags([1 , -2 , 1], [ -1 , 0 , 1], shape = (n, n), dtype=None) / h**2

    # 2D Laplacian : L = I_y (x) Dx + Dy (x) I_x
    Ix = sp.eye(n)
    Iy = sp.eye(n)
    L = sp.kron(Iy, Dx) + sp.kron(Dy, Ix)

    F = f(X, Y)
    F = F.flatten()

    # Exact solution / Dirichlet BC
    U_exact = exact(X, Y)
    u_bc = U_exact.flatten()

    # Find boundary points
    boundary = np.zeros((n, n), dtype=bool)

    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True

    boundary_idx = np.where(boundary.flatten())[0]

    # Enforce Dirichlet BC
    L = L.tolil()

    for k in boundary_idx:
        L.rows[k] = [k]
        L.data[k] = [1.0]
        if(homogeneous_bndry):
            F[k] = 0
        else:
            F[k] = u_bc[k]

    L = L.tocsr()

    Nu = X.size
    def Amult(v):
        u = v[:Nu]
        q = v[Nu:]

        if fast:
            spread = S_fast(q, X, Y, x_ib, y_ib, a).ravel()
        else:
            spread = S(q, X, Y, x_ib, y_ib, a).ravel()

        # No spreading contribution on Dirichlet boundary rows
        spread[boundary_idx] = 0.0

        top = L @ u - spread
        if fast:
            bottom = J_fast(u.reshape(X.shape), X, Y, x_ib, y_ib, a, h)
        else:
            bottom = J(u.reshape(X.shape), X, Y, x_ib, y_ib, a, h)

        return np.concatenate([top, bottom])

    A = LinearOperator(
        shape=(Nu + N_ib, Nu + N_ib),
        matvec=Amult,
        dtype=float
    )

    g = exact(x_ib, y_ib)

    rhs = np.concatenate([F, g])

    solution, info = gmres(
        A, rhs,
        restart=1000,
        rtol=1e-8,
        maxiter=1000
    )

    if info != 0:
        raise RuntimeError(f"GMRES did not converge (info={info}).")

    u_sol = solution[:Nu].reshape(X.shape)
    q_sol = solution[Nu:]

    return x, y, x_ib, y_ib, u_sol, q_sol

@dataclass
class BenchmarkResult:
    """Timing and paired solution-error data for one IB boundary problem."""

    slow_seconds: np.ndarray
    fast_seconds: np.ndarray
    relative_solution_error: np.ndarray


def benchmark_problem(f, exact, N, homogeneous_bndry, replications):
    """Time slow and fast IB solves and compare paired numerical solutions."""
    slow_seconds = np.empty(replications)
    fast_seconds = np.empty(replications)
    relative_solution_error = np.empty(replications)

    for replication in range(replications):
        start = perf_counter()
        *_, u_slow, _ = IB_finite_diff(
            f, N, 0.0, 2 * np.pi, exact,
            homogeneous_bndry=homogeneous_bndry, fast=False,
        )
        slow_seconds[replication] = perf_counter() - start

        start = perf_counter()
        *_, u_fast, _ = IB_finite_diff(
            f, N, 0.0, 2 * np.pi, exact,
            homogeneous_bndry=homogeneous_bndry, fast=True,
        )
        fast_seconds[replication] = perf_counter() - start

        denominator = np.linalg.norm(u_slow.ravel())
        relative_solution_error[replication] = (
            np.linalg.norm((u_fast - u_slow).ravel()) / denominator
            if denominator != 0.0 else np.linalg.norm((u_fast - u_slow).ravel())
        )

    return BenchmarkResult(slow_seconds, fast_seconds, relative_solution_error)


def plot_runtimes(result, problem_name, output_path):
    """Save a four-line runtime plot: two runs and two mean-runtime lines."""
    repetitions = np.arange(1, result.slow_seconds.size + 1)
    slow_mean = result.slow_seconds.mean()
    fast_mean = result.fast_seconds.mean()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(repetitions, result.slow_seconds, "o-", label="Slow runtime")
    ax.plot(repetitions, result.fast_seconds, "s-", label="Fast runtime")
    ax.axhline(slow_mean, color="C0", linestyle="--", label="Slow average")
    ax.axhline(fast_mean, color="C1", linestyle="--", label="Fast average")
    ax.set(
        xlabel="Replication",
        ylabel="Runtime (s)",
        title=f"IB finite-difference runtime: {problem_name}",
        xticks=repetitions,
    )
    ax.grid(alpha=0.3)
    ax.margins(y=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark slow and fast immersed-boundary finite differences."
    )
    parser.add_argument(
        "-N", "--grid-points", type=int, default=32,
        help="Number of grid intervals in each coordinate direction (default: 32).",
    )
    parser.add_argument(
        "-r", "--replications", type=int, default=3,
        help="Number of paired slow/fast replications per problem (default: 3).",
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=Path("."),
        help="Directory for PNG plots and the NPZ timing/error data (default: current directory).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.grid_points < 2:
        raise ValueError("--grid-points must be at least 2.")
    if args.replications < 1:
        raise ValueError("--replications must be at least 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    exact = lambda x, y: np.cos(2 * x) * np.sin(2 * y)
    original_forcing = lambda x, y: -8 * np.cos(2 * x) * np.sin(2 * y)
    problem3_forcing = lambda x, y: np.zeros_like(x, dtype=float)

    problems = (
        ("original boundary problem", "original", original_forcing, False),
        ("Problem 3 modified problem", "problem3", problem3_forcing, True),
    )
    results = {}
    for title, stem, forcing, homogeneous_bndry in problems:
        print(f"Benchmarking {title} (N={args.grid_points})...")
        result = benchmark_problem(
            forcing, exact, args.grid_points, homogeneous_bndry, args.replications
        )
        results[stem] = result
        plot_path = args.output_dir / f"ib_runtime_{stem}_N{args.grid_points}.png"
        plot_runtimes(result, title, plot_path)
        print(f"  slow mean: {result.slow_seconds.mean():.6f} s")
        print(f"  fast mean: {result.fast_seconds.mean():.6f} s")
        print("  paired relative solution errors:", result.relative_solution_error)
        print("  mean relative solution error:", np.mean(result.relative_solution_error))
        print(f"  plot: {plot_path}")

    data_path = args.output_dir / f"ib_timing_N{args.grid_points}.npz"
    np.savez(
        data_path,
        original_slow_seconds=results["original"].slow_seconds,
        original_fast_seconds=results["original"].fast_seconds,
        original_relative_solution_error=results["original"].relative_solution_error,
        problem3_slow_seconds=results["problem3"].slow_seconds,
        problem3_fast_seconds=results["problem3"].fast_seconds,
        problem3_relative_solution_error=results["problem3"].relative_solution_error,
    )
    print(f"Saved timing and paired-error arrays: {data_path}")


if __name__ == "__main__":
    main()
