import numpy as np
import polars as pl
import pandas as pd
import scipy.stats as st
from tqdm import tqdm

def generate_stock_path(S_0: float, nbr_gen: int, time_obs: int, timeline: np.ndarray, delta_t: float, mu: float, sigma: float, phi: float = 0) -> tuple:
    """
    Simulates stock price paths using a geometric Brownian motion model with optional drift adjustment.

    :param S_0: float
        Initial stock price.
    :param nbr_gen: int
        Number of stock price paths to generate.
    :param time_obs: int
        Number of time observations for the simulation.
    :param timeline: np.array
        A sequence of time points corresponding to the observations.
    :param delta_t: float
        Time increment between observations.
    :param mu: float
        Expected rate of return -> r.
    :param sigma: float
        Volatility of the stock (diffusion coefficient).
    :param phi: float, optional (default=0)
        Adjustment factor for Brownian motion increments. Defaults to 0 (no adjustment).
    :return: tuple
        - stock_price: polars.DataFrame
            DataFrame containing the simulated stock price paths for each time point.
        - dW_sum: numpy.ndarray
            Array of cumulative Brownian motion increments for each path, adjusted by `phi` if specified.
    """
    stock_price = pl.DataFrame({"S_0.0": pl.Series([S_0] * nbr_gen)})
    dW_sum = np.zeros(nbr_gen)
    for t in range(time_obs):
        #brownian increment, same as Wt+1-Wt
        dW = st.norm.rvs(loc = phi * delta_t, scale=np.sqrt(delta_t), size = nbr_gen)

        #add observations
        stock_price = stock_price.with_columns(
            (pl.col(f"S_{timeline[t]}")*(1+mu*delta_t+sigma*dW)).alias(f"S_{timeline[t+1]}")
        )

        if phi != 0:
            dW_sum += dW

    stock_price = stock_price.with_columns(
        pl.Series("likelihood_ratio", np.exp(0.5*delta_t*time_obs*phi**2-phi*dW_sum))
    )

    return stock_price, dW_sum

def generate_CI(Ns: np.ndarray, S_0: float, K: float, M: int, T: float, timeline: np.ndarray, dt: float, r: float, vol: float, alpha: float) -> dict:
    """
    Generates confidence intervals for the payoff of an option by simulating stock price paths

    :param Ns: np.ndarray
        Array of different sample sizes for which to compute the confidence interval.
    :param S_0: float
        Initial stock price.
    :param K: float
        Strike price of the option.
    :param M: int
        Number of time steps for the simulation.
    :param T: float
        Total time to maturity.
    :param timeline: np.ndarray
        Array of time points corresponding to the observations.
    :param dt: float
        Time increment between observations.
    :param r: float
        Risk-free interest rate.
    :param vol: float
        Volatility of the stock -> sigma.
    :param alpha: float
        Significance level for the confidence interval.
    :return: dict
        Dictionary where keys are sample sizes and values are the corresponding confidence intervals.
    """
    CI = {}
    for N in tqdm(Ns):
        stock_prices,_ = generate_stock_path(S_0=S_0, nbr_gen=N, time_obs=M, timeline=timeline, delta_t=dt, mu=r, sigma=vol)

        final_stock = f"S_{T}"

        stock_prices = stock_prices.with_columns(
            (np.maximum(pl.col(final_stock) - K, 0) * np.exp(-r * T)).alias("payoff")
        )

        stock_prices = stock_prices.with_columns(
            (pl.col("payoff") * pl.col("likelihood_ratio")).alias("payoff_weighted")
        )
        vol_Z = (stock_prices.select("payoff_weighted")).std(ddof=1).item()
        ci_N = st.norm.ppf(1 - alpha / 2) * vol_Z / np.sqrt(N)
        CI.update({f"CI({N})":ci_N})
    return CI

def generate_numerical_prices(iterations,Ns,S_0, K, M, T, timeline, dt, r, vol, alpha = 0.05, phi=0, U=np.inf):
    """

    :param iterations:
    :param Ns:
    :param S_0:
    :param K:
    :param M:
    :param T:
    :param timeline:
    :param dt:
    :param r:
    :param vol:
    :param alpha:
    :param phi:
    :param U:
    :return:
    """
    # Storage for results over iterations
    column_names = [f"sample_size_{N}" for N in Ns]
    numerical_prices = pd.DataFrame(columns=column_names)
    confidence_intervals = pd.DataFrame(columns=column_names)
    for iter in tqdm(iterations, ascii=True, desc="numerical price generation"):
        numerical_prices_iter = []
        confidence_interval_iter = []
        for N in Ns:
            stock_prices,_ = generate_stock_path(S_0=S_0, nbr_gen=N, time_obs=M, timeline=timeline, delta_t=dt, mu=r,
                                               sigma=vol,phi = phi)

            if U != np.inf:
                stock_prices = stock_prices.with_columns(
                    pl.max_horizontal(pl.exclude("likelihood_ratio")).alias("S_max")
                )
                stock_prices = stock_prices.with_columns(
                    (pl.when(pl.col("S_max") <= U).then(1).otherwise(0)).alias("condition")
                )
            else:
                stock_prices = stock_prices.with_columns(
                    pl.Series("condition",  [1] * N)
                )

            final_stock = f"S_{T}"

            stock_prices = stock_prices.with_columns(
                (np.maximum(pl.col(final_stock) - K, 0)* np.exp(-r * T)).alias("payoff")
            )

            stock_prices = stock_prices.with_columns(
                (pl.col("payoff") * pl.col("likelihood_ratio") * pl.col("condition")).alias("payoff_weighted")
            )

            numerical_prices_iter.append((stock_prices.select("payoff_weighted")).mean().item())

            #Confidence intervals building
            vol_Z = (stock_prices.select("payoff_weighted")).std(ddof=1).item()
            ci_N = st.norm.ppf(1 - alpha / 2) * vol_Z / np.sqrt(N)
            confidence_interval_iter.append(ci_N)
        #Add list of values at the end of my dataframe
        numerical_prices.loc[len(numerical_prices)] = numerical_prices_iter
        confidence_intervals.loc[len(confidence_intervals)] = confidence_interval_iter
    return numerical_prices, confidence_intervals

def generate_controled_path(S_0, nbr_gen, time_obs, timeline, delta_t, mu, sigma, s_min, s_max, zeta_interpolator):
    """

    :param S_0:
    :param nbr_gen:
    :param time_obs:
    :param timeline:
    :param delta_t:
    :param mu:
    :param sigma:
    :param s_min:
    :param s_max:
    :param zeta_interpolator:
    :return:
    """
    stock_price = pl.DataFrame({"S_0.0": pl.Series([S_0] * nbr_gen)})
    likelihood_ratio = np.zeros(nbr_gen)
    for t in range(time_obs-1):  # Ensure t + 1 does not exceed bounds
        # Brownian increment, same as Wt+1 - Wt
        stock_val = stock_price.select([f"S_{timeline[t]}"]).to_numpy()
        dW = np.empty(nbr_gen)
        for n in range(nbr_gen):
            s = stock_val[n]
            if s >= s_min and s <= s_max:
                zeta_control = zeta_interpolator((s, timeline[t])).flatten()
            elif s < s_min:
                zeta_control = zeta_interpolator((s_min, timeline[t])).flatten()
            else: # s > s_max
                zeta_control = zeta_interpolator((s_max, timeline[t])).flatten()

            dW[n] = st.norm.rvs(loc=delta_t*zeta_control.flatten(), scale=np.sqrt(delta_t), size=1).flatten()

            likelihood_ratio[n] += delta_t/2*(zeta_control.flatten())**2 - zeta_control.flatten()*dW[n]

        # Add observations
        stock_price = stock_price.with_columns(
            (pl.col(f"S_{timeline[t]}") * (1 + mu * delta_t + sigma * dW)).alias(f"S_{timeline[t + 1]}")
        )

    stock_price = stock_price.with_columns(
        pl.Series("likelihood_ratio", np.exp(likelihood_ratio))
    )

    return stock_price, likelihood_ratio