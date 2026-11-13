import numpy as np
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.sparse.linalg import factorized
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

class time_stepper():
    def __init__(self, f, N, min, max, exact):
        """Solve u_t = u_xx + u_yy + f with Dirichlet boundary conditions."""
        if N < 3:
            raise ValueError("N must each be at least 3.")

        h = (max - min) / (N-1)

        x = np.linspace(min, max, N)
        y = np.linspace(min, max, N)

        # Interior grid points
        xi = x[1:-1]
        yi = y[1:-1]

        X, Y = np.meshgrid(xi, yi)

        n = N - 2

        Dx = sp.diags([1 , -2 , 1], [ -1 , 0 , 1], shape = (n, n), dtype=None) / h**2
        Dy = sp.diags([1 , -2 , 1], [ -1 , 0 , 1], shape = (n, n), dtype=None) / h**2

        # 2D Laplacian : L = I_y (x) Dx + Dy (x) I_x
        Ix = sp.eye(n)
        Iy = sp.eye(n)
        L = sp.kron(Iy, Dx) + sp.kron(Dy, Ix)

        def F(t):
            return f(X, Y, t).flatten()
        
        # Initial condition from exact solution
        u0 = exact(X, Y, 0).flatten()

        self.x = x
        self.y = y
        self.L = L
        self.F = F
        self.u0 = u0
        self.h = h

    def getGrid(self):
        return self.x, self.y

    def getH(self):
        return self.h

    def _forward_euler(self, dt, t_init, t_max):
        u = self.u0.copy()
        t_vals = np.arange(t_init, t_max, dt)
        for t in t_vals:
            u = u + dt * (self.L @ u + self.F(t))

        return u

    def _backward_euler(self, dt, t_init, t_max):
        u = self.u0.copy()
        t_vals = np.arange(t_init+dt, t_max + dt/2, dt)

        I = sp.eye(self.L.shape[0], format='csr')
        A = I - dt * self.L
        solve = factorized(A.tocsc())
        
        for t in t_vals:
            b = u + dt*self.F(t)
            u = solve(b)

        return u

    def _crank_nicolson(self, dt, t_init, t_max):
        u = self.u0.copy()
        t_vals = np.arange(t_init+dt, t_max + dt/2, dt)

        I = sp.eye(self.L.shape[0], format='csr')
        A = I - (dt / 2) * self.L
        solve = factorized(A.tocsc())

        B = I + (dt/2) * self.L
        
        for n in range(len(t_vals)-1):
            t_n = t_vals[n]
            t_next = t_vals[n+1]
            b = B @ u + (dt / 2) * (self.F(t_n)+ self.F(t_next))
            u = solve(b)

        return u

    def _RK2(self, dt, t_init, t_max):
        u = self.u0.copy()
        t_vals = np.arange(t_init, t_max, dt)
        for t in t_vals:
            u_half = u + (dt / 2) * (self.L @ u + self.F(t))
            u = u + dt * (self.L @ u_half + self.F(t + dt/2))


        return u

    def _BDF2(self, dt, t_init, t_max):
        I = sp.eye(self.L.shape[0], format='csr')
        u_prev = self.u0.copy()
        # backward Euler to get first u
        A = I - dt * self.L
        b = u_prev + dt*self.F(t_init + dt)
        u = spsolve(A, b)

        t_vals = np.arange(t_init + 2 * dt, t_max + dt/2, dt)

        A = 3 * I - 2 * dt * self.L
        solve = factorized(A.tocsc())
        
        for t in t_vals:
            b = 4 * u - u_prev + 2 * dt * self.F(t)
            u_next = solve(b)
            u_prev = u
            u = u_next

        return u

    def _RK45(self, dt, t_init, t_max):
        def rhs(t, u):
            return self.L @ u + self.F(t)

        sol = solve_ivp(rhs, (t_init, t_max), self.u0, method="RK45", 
                        max_step=dt,rtol=1e-6,atol=1e-8)

        if not sol.success:
            raise RuntimeError(sol.message)

        return sol.y[:, -1]

    # Single function that can call any of the other functions based on a defined method
    def time_step(self, dt, t_init, t_max, method):
        methods = {
            "forward_euler": self._forward_euler,
            "backward_euler": self._backward_euler,
            "crank_nicolson": self._crank_nicolson,
            "rk2": self._RK2,
            "bdf2": self._BDF2,
            "rk45": self._RK45
        }

        if method not in methods:
            raise ValueError(f"Unknown method '{method}'. Choose from {list(methods)}")

        return methods[method](dt, t_init, t_max)

def relative_error(numerical, exact):
    """Compute the relative error between an approximate and exact value."""
    return np.linalg.norm(numerical - exact) / np.linalg.norm(exact)


def main():
    def exact(x, y, t):
        return np.cos(10*t) * np.sin(x) * np.sin(y)

    def f(x, y, t):
        return (2*np.cos(10*t) - 10*np.sin(10*t)) * np.sin(x) * np.sin(y)

    N = 64

    stepper = time_stepper(f, N, 0, 2*np.pi, exact)

    x, y = stepper.getGrid()

    t_init = 0
    t_max = 2*np.pi
    dt = (t_max-t_init) / 30

    # returns a flattened vector(without boundary conditions)
    u_numerical = stepper.time_step(dt, t_init, t_max, "crank_nicolson")

    # reconstruct
    U = np.zeros((N, N))
    U[1:-1, 1:-1] = u_numerical.reshape(N-2, N-2)

    fig_plotly = go.Figure()

    fig_plotly.add_trace(
        go.Surface(
            x=x, y=y, z=U,
            colorscale="Plasma",
            colorbar=dict(title="u(x, y)"),
            opacity=0.85,
            name="Numerical solution",
            showscale=True,
        ),
    )

    fig_plotly.update_layout(
        title="Solution to forced heat equation",
        width=1100, height=650,
        margin=dict(b=90),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.12, yanchor="top"),
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="u(x, y)",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.55),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
        ),
    )
    fig_plotly.show()


if __name__ == "__main__":
    main()