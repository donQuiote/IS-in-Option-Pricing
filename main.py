import numpy as np
import polars as pl
from grapher import plot_MC_Analytical, plot_paths
from generation import generate_numerical_prices, generate_stock_path
from helpers import generate_latex_table, generate_summary_table, analytical_call_price
from adaptive_strategy import adaptive_optimal_r

#Interest rate
r = .05
#stock volatility
vol = 0.51

#Option maturity
T = 0.2
#initial asset price
S0 = 100
#strike price
K = 120
#Subintervals
M = 100
#Smapling size
nbr_Ns = 30
Ns = np.logspace(1,6,num=nbr_Ns,dtype=int) #np.ndarray

# iteration of each process
nbr_iterations = 20
iterations = np.arange(0,nbr_iterations) #np.ndarray

# We will assume t = 0 for simplicity
dt = T/M
timeline = np.linspace(0,T,num=M+1,endpoint=True)
analytical_price = analytical_call_price(S_0=S0,K=K,T=T,r=r,sigma=vol)
print(analytical_price)
#%%
##################
# 1.
##################
#Generate the paths and plot
N = 10
stock_prices,_ = generate_stock_path(S_0=S0, nbr_gen=N, time_obs=M, timeline=timeline, delta_t=dt, mu=r, sigma=vol,phi = 0)
stock_paths = stock_prices.select(pl.exclude("likelihood_ratio")).to_numpy()
plot_paths(stock_paths=stock_paths, timeline=timeline, N=N, K=K, filename="images/stock_paths")

#%%
#Generate multiple iterations of various path and compute the price and Confidence intervals
numerical_prices, confidence_intervals = generate_numerical_prices(iterations=iterations,Ns=Ns,S_0=S0, K=K, M=M, T=T, timeline=timeline, dt=dt, r=r, vol=vol)
numerical_prices_avg =  numerical_prices.mean()
numerical_prices_std = numerical_prices.std()
analytical_price = analytical_call_price(S_0=S0,K=K,T=T,r=r,sigma=vol)
plot_MC_Analytical(analytical_price=analytical_price, numerical_prices_avg=numerical_prices_avg,  numerical_prices_std=numerical_prices_std, sample_sizes=Ns, nbr_iterations = nbr_iterations, confidence_level=0.95, filename = "images/CMC_CO_estimation")
latex_table = generate_latex_table(numerical_prices, Ns,mean_txt="Mean numerical price", std_txt="estimator's std dev.")
print(latex_table)
latex_table_CI = generate_latex_table(confidence_intervals, Ns, mean_txt="Mean CI size", std_txt="sdt dev. in CI size")
print(latex_table_CI)

##################
# 2.
##################
#%%
r_tilde = np.log(K/S0)/T
phi = (r_tilde-r)/vol
#%%

numerical_prices_IS, confidence_intervals_IS = generate_numerical_prices(iterations=iterations,Ns=Ns,S_0=S0, K=K, M=M, T=T, timeline=timeline, dt=dt, r=r, vol=vol, phi=phi)
numerical_prices_avg_IS =  numerical_prices_IS.mean()
numerical_prices_std_IS = numerical_prices_IS.std()
plot_MC_Analytical(analytical_price=analytical_price, numerical_prices_avg=numerical_prices_avg_IS,  numerical_prices_std=numerical_prices_std_IS, sample_sizes=Ns, nbr_iterations = nbr_iterations, confidence_level=0.95, filename = "images/MC_CO_estimation_IS")
latex_table_IS = generate_latex_table(numerical_prices_IS, Ns,mean_txt="Mean numerical price", std_txt="estimator's std dev.")
print(latex_table_IS)
latex_table_CI_IS = generate_latex_table(confidence_intervals_IS, Ns, mean_txt="Mean CI size", std_txt="sdt dev. in CI size")
print(latex_table_CI_IS)

#%%
summary = generate_summary_table(Ns,numerical_prices_std_IS,numerical_prices_std, confidence_intervals_IS,confidence_intervals, numerical_prices_avg, numerical_prices_avg_IS, analytical_price)
print(summary)
#%%
##################
# 3.
##################
r_star, phi_star, price_est_star, price_est_vol_star = adaptive_optimal_r(S0=S0 ,r_0=r, r_tilde=r_tilde, vol=vol,K=K , M=M, dt=dt, timeline=timeline,T=T)
print(f"after adaptive optimization, the best r_tilde is {r_star}, we obtain the price of {price_est_star} and volatility {price_est_vol_star**2}")
#%%
numerical_prices_star, confidence_intervals_star = generate_numerical_prices(iterations=iterations,Ns=Ns,S_0=S0, K=K, M=M, T=T, timeline=timeline, dt=dt, r=r, vol=vol, phi=phi_star)
numerical_prices_avg_star =  numerical_prices_star.mean()
numerical_prices_std_star = numerical_prices_star.std()
analytical_price = analytical_call_price(S_0=S0,K=K,T=T,r=r,sigma=vol)
plot_MC_Analytical(analytical_price=analytical_price, numerical_prices_avg=numerical_prices_avg_star,  numerical_prices_std=numerical_prices_std_star, sample_sizes=Ns, nbr_iterations = nbr_iterations, confidence_level=0.95, filename="images/MC_CO_Optimal_r_tilde")
latex_table_star = generate_latex_table(numerical_prices_star, Ns,mean_txt="Mean numerical price", std_txt="estimator's std dev.")
print(latex_table_star)
latex_table_CI_star = generate_latex_table(confidence_intervals_star, Ns, mean_txt="Mean CI size", std_txt="sdt dev. in CI size")
print(latex_table_CI_star)

summary_star_vs_CMC = generate_summary_table(Ns,numerical_prices_std_star,numerical_prices_std, confidence_intervals_star,confidence_intervals, numerical_prices_avg, numerical_prices_avg_star, analytical_price)
print(summary_star_vs_CMC)

summary_star_vs_tilde = generate_summary_table(Ns,numerical_prices_std_star,numerical_prices_std_IS, confidence_intervals_star,confidence_intervals_IS, numerical_prices_avg_IS, numerical_prices_avg_star, analytical_price)
print(summary_star_vs_tilde)