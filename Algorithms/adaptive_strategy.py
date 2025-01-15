import polars as pl
import scipy.stats as st
import numpy as np
from Algorithms.generation import generate_stock_path
from scipy.optimize import minimize


def adaptive_optimal_r(S0: float, r_0: float, r_tilde: float, vol: float, K: float, M: int, dt: float,
                       timeline: np.ndarray, T: float, U: float = np.inf, N_bar: int = 100,
                       gamma: float = 2, tol: float = 1.5E-4, alpha: float = 0.05) -> tuple:
    """
    Computes the optimal adjusted rate of return (`r_star`) using an adaptive simulation approach.

    The method iteratively refines the adjustment parameter `phi` to minimize the variance of the weighted payoff
    and ensures convergence to a target precision within the specified tolerance.

    The algorithm is described in the report attached to the code

    Parameters
    ----------
    S0 : float
        Initial stock price.
    r_0 : float
        Initial expected rate of return.
    r_tilde : float
        Initial adjustment to the rate of return (`r`).
    vol : float
        Volatility of the stock price.
    K : float
        Strike price of the option.
    M : int
        Number of time observations (steps) in the simulation.
    dt : float
        Time increment between consecutive observations.
    timeline : np.ndarray
        Sequence of time points corresponding to the observations.
    T : float
        Total time to maturity (in years).
    U : float, optional
        Upper barrier for stock price paths. Default is `np.inf` (no barrier).
    N_bar : int, optional
        Initial number of simulations. Default is 100.
    gamma : float, optional
        Growth factor for the number of simulations in each iteration. Default is 2.
    tol : float, optional
        Precision tolerance for the stopping criterion of the simulation. Default is 1.5E-4.
    alpha : float, optional
        Significance level for confidence intervals. Default is 0.05.

    Returns
    -------
    tuple
        A tuple containing:
        - r_star (float): The optimal adjusted rate of return.
        - phi (float): The final adjustment parameter.
        - mu_Z (float): The mean of the weighted payoff.
        - vol_Z (float): The standard deviation of the weighted payoff.
    """
    #Initialization
    phi = (r_tilde - r_0) / vol
    c = st.norm.ppf(1 - alpha / 2)
    vol_Z = np.inf
    N = N_bar / gamma
    mu_Z = None

    while (vol_Z*c)/np.sqrt(N)>tol and N<20E6:
        N=int(gamma*N)
        stock_price, dW_sum = generate_stock_path(S_0=S0, nbr_gen=N, time_obs=M, timeline=timeline, delta_t=dt, mu=r_0,
                                              sigma=vol, phi=phi)

        if U != np.inf:
            stock_price = stock_price.with_columns(
                pl.max_horizontal(pl.exclude("likelihood_ratio")).alias("S_max")
            )
            stock_price = stock_price.with_columns(
                (pl.when(pl.col("S_max") <= U).then(1).otherwise(0)).alias("condition")
            )
        else:
            stock_price = stock_price.with_columns(
                pl.Series("condition", [1] * N)
            )

        final_stock = f"S_{T}"

        stock_price = stock_price.with_columns(
            (np.maximum(pl.col(final_stock) - K, 0)* np.exp(-r_0 * T)).alias("payoff")
        )

        stock_price = stock_price.with_columns(
            (pl.col("payoff") * pl.col("likelihood_ratio")* pl.col("condition")).alias("payoff_weighted")
        )

        mu_Z = (stock_price.select("payoff_weighted")).mean().item()

        vol_Z =  (stock_price.select("payoff_weighted")).std(ddof=1).item()

        def phi_star_finder(phi_star, stock_price, dW_sum, phi):
            stock_price = stock_price.with_columns(
                pl.Series("correction", np.exp(0.5 * dt * M * (phi ** 2+phi_star**2) - (phi+phi_star) * dW_sum))
            )
            stock_price = stock_price.with_columns(
                (pl.col("payoff").pow(2) * pl.col("likelihood_ratio")*pl.col("correction")).alias("to_minimize")
            )
            value = (stock_price.select("to_minimize")).mean().item()
            return value

        phi = minimize(phi_star_finder, x0=1., args=(stock_price,dW_sum, phi), tol=1e-6).x.item()

        print(f"the optimal r tilde is {phi*vol+r_0}")
        print(f"{(vol_Z*c)/np.sqrt(N)},{N}")

    r_star = phi*vol+r_0

    return r_star, phi, mu_Z, vol_Z