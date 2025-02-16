import numpy as np
from scipy.linalg import solve_banded

def crank_nicholson_pde_solver(V_matrix : np.ndarray, dt : float, ds : float, s_set : np.ndarray, M_tilde : int, P : int, vol : float, r : float):
    """
    Solves a stochastic differential equation (SDE) backward in time using the Crank-Nicholson method.

    Parameters
    ----------
    V_matrix : np.ndarray
        A 2D array representing the discretized value function across space and time. The array is updated in-place
        during the backward time iteration.
    dt : float
        Time step size for the backward iteration.
    ds : float
        Space step size for the discretization of the spatial variable (e.g., stock price).
    s_set : np.ndarray
        Array of spatial grid points corresponding to the stock price levels.
    M_tilde : int
        Total number of time steps in the discretized time grid.
    P : int
        Total number of spatial grid points in the discretized space grid.
    vol : float
        Volatility of the underlying asset.
    r : float
        Risk-free interest rate.

    Returns
    -------
    np.ndarray
        The updated value function `V_matrix` after solving the PDE backward in time. Each column represents the value function
        at a specific time step, and each row corresponds to a specific spatial point.
    """
    # As descibed in the report build the M1 matrix
    d_m = np.array([1 / dt + 2 * ((s * vol) ** 2) / (4 * (ds ** 2)) for s in s_set[1:-1]])
    u_m = np.array([-(r * s) / (4 * ds) - (s * vol) ** 2 / (4 * (ds ** 2)) for s in s_set[1:-2]])
    l_m = np.array([(r * s) / (4 * ds) - (s * vol) ** 2 / (4 * (ds ** 2)) for s in s_set[2:-1]])

    # Constructing the tridiagonal matrix
    # M_m = np.diag(d_m, k=0) + np.diag(u_m, k=1) + np.diag(l_m, k=-1)

    M_m = np.zeros((3, P - 2))
    M_m[0, 1:] = u_m
    M_m[1, :] = d_m
    M_m[2, :-1] = l_m
    # M_m = np.diag(d_m, k=0) + np.diag(u_m, k=1) + np.diag(l_m, k=-1)

    # As descibed in the report build the M2 matrix
    d_mp1 = np.array([1 / dt - 2 * ((s * vol) ** 2) / (4 * (ds ** 2)) for s in s_set[1:-1]])
    u_mp1 = np.array([(r * s) / (4 * ds) + (s * vol) ** 2 / (4 * (ds ** 2)) for s in s_set[1:-2]])
    l_mp1 = np.array([-(r * s) / (4 * ds) + (s * vol) ** 2 / (4 * (ds ** 2)) for s in s_set[2:-1]])

    M_mp1 = np.diag(d_mp1, k=0) + np.diag(u_mp1, k=1) + np.diag(l_mp1, k=-1)
    # Iterate backward in time
    for j in range(M_tilde - 2, -1, -1):
        b_m = np.zeros(P - 2)
        b_m[0] = 0
        b_m[-1] = (-(r * s_set[-1]) / (4 * ds) - (s_set[-1] * vol) ** 2 / (4 * ds ** 2)) * V_matrix[-1, j]

        b_mp1 = np.zeros(P - 2)
        b_mp1[0] = 0
        b_mp1[-1] = ((r * s_set[-1]) / (4 * ds) - (s_set[-1] * vol) ** 2 / (4 * ds ** 2)) * V_matrix[-1, j + 1]

        y = np.dot(M_mp1, V_matrix[1:-1, j + 1]) + b_mp1 - b_m

        # Solve the banded system
        V_matrix[1:-1, j] = solve_banded((1, 1), M_m, y)

    return V_matrix


def zeta_control_generator(P : int, M_tilde : int, V_matrix : np.ndarray, s_set : np.ndarray, ds : float, vol : float):
    """
    Generates the control parameter `zeta` used in a stochastic control framework.

    This function computes the `zeta` control matrix based on the value function `V_matrix`
    generated by the Crank-Nicholson solver. The control is derived from the forward differences
    of the value function with respect to the spatial grid.

    Parameters
    ----------
    P : int
        Number of spatial grid points (e.g., stock price levels).
    M_tilde : int
        Number of time steps in the discretized time grid.
    V_matrix : np.ndarray
        A 2D array representing the value function computed from the Crank-Nicholson solver.
        The rows correspond to spatial points, and the columns to time points.
    s_set : np.ndarray
        Array of spatial grid points corresponding to the stock price levels.
    ds : float
        Space step size for the discretization of the spatial variable.
    vol : float
        Volatility of the underlying asset.

    Returns
    -------
    np.ndarray
        A 2D array `zeta_forward` of shape `(P, M_tilde)` representing the control parameter `zeta`
        at each spatial and temporal point. The control is adjusted for boundary conditions.
    """
    zeta_forward = np.zeros((P, M_tilde))
    for j in range(M_tilde):
        forward_diff = np.diff(np.abs(V_matrix[:, j]), prepend=V_matrix[1, j]) / ds

        zeta_forward[:, j] = forward_diff * vol * (s_set / np.abs(V_matrix[:, j]))

    # Prolongate the control
    zeta_forward[0, :] = zeta_forward[1, :]
    # Set control to 0 at T
    zeta_forward[:, -1] = np.zeros(P)
    # Set s_max
    zeta_forward[-1, :] = zeta_forward[-2, :]

    return zeta_forward