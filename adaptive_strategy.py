import polars as pl
import scipy.stats as st
import numpy as np
from generation import generate_stock_path
from scipy.optimize import minimize


def adaptive_optimal_r(S0 ,r_0, r_tilde, vol,K , M, dt, timeline,T, U=np.inf, N_bar=100, gamma=2, tol=1.5E-4, alpha = 0.05):
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