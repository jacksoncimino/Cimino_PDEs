# Homework 2: Problem 4 benchmark

`p4.py` benchmarks the immersed-boundary finite-difference solver with its
full-grid implementation (`fast=False`) and its cutoff-based implementation
(`fast=True`). It runs both the original boundary-value problem and Problem 3's
modified homogeneous-outer-boundary problem.

## Run

From this directory, run:

```powershell
py p4.py --grid-points 64 --replications 5 --output-dir results
```

Options:

- `--grid-points` / `-N`: number of grid intervals in each coordinate direction.
- `--replications` / `-r`: paired fast/slow runs for each problem.
- `--output-dir` / `-o`: directory in which to save results (defaults to the current directory).

The script prints mean runtimes and the paired relative solution-error array.
It also saves one runtime plot per problem and an `.npz` archive with
the fast and slow runtime arrays plus the per-replication relative errors.
