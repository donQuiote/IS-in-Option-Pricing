import numpy as np
import polars as pl
import pandas as pd
import scipy.stats as st
from tqdm import tqdm

def generate_stock_path(S_0, nbr_gen, time_obs, timeline, delta_t,mu,sigma, phi = 0):
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

def generate_CI(Ns,S_0, K, M, T, timeline, dt, r, vol, alpha):
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