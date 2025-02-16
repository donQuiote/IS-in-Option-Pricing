import numpy as np
from Algorithms.generation import generate_numerical_prices, generate_stock_path
from Algorithms.adaptive_strategy import adaptive_optimal_r
from Utils.grapher import plot_MC_Analytical
from Utils.helpers import generate_latex_table, generate_summary_table, analytical_Up_Out_call_price

#Interest rate
r = .1
#stock volatility
vol = 0.3

#Option maturity
T = 0.2
#initial asset price
S0 = 100
#strike price
K = 150
#Up strike price
U = 200
#U = 170
#Subintervals
M = 1000
#Smapling size
nbr_Ns = 20
Ns = np.logspace(2,6,num=nbr_Ns,dtype=int)

# iteration of each process
nbr_iterations = 12
iterations = np.arange(0,nbr_iterations)

# We will assume t = 0 for simplicity
dt = T/M
timeline = np.linspace(0,T,num=M+1,endpoint=True)
#%%
#Up_Out_price = analytical_Up_Out_call_price(S0, K, U, T, r, vol)
#print(Up_Out_price)
#%%
##################
# 1.
##################
numerical_prices, confidence_intervals = generate_numerical_prices(iterations=iterations,Ns=Ns,S_0=S0, K=K, M=M, T=T, timeline=timeline, dt=dt, r=r, vol=vol, generate=generate_stock_path, U=U)
numerical_prices_avg =  numerical_prices.mean()
numerical_prices_std = numerical_prices.std()
plot_MC_Analytical(analytical_price=Up_Out_price, numerical_prices_avg=numerical_prices_avg, numerical_prices_std=numerical_prices_std, sample_sizes=Ns, nbr_iterations = nbr_iterations, confidence_level=0.95, filename="../Graphs/MC_UOCO_estimation", title="Crude Monte Carlo Up and Out call price estimation in function of sample size")
latex_table = generate_latex_table(numerical_prices, Ns,mean_txt="Mean numerical price", std_txt="estimator's std dev.")
print(latex_table)
latex_table_CI = generate_latex_table(confidence_intervals, Ns, mean_txt="Mean CI size", std_txt="sdt dev. in CI size")
print(latex_table_CI)

#%%
##################
# 2.
##################
N_bar = 1000
gamma = 5
tol = 5E-4
alpha = 0.05
r_tilde = np.log(K/S0)/T
r_star, phi_star, price_est_star, price_est_vol_star = adaptive_optimal_r(S0=S0 ,r_0=r, r_tilde=r_tilde, vol=vol,K=K , M=M, dt=dt, timeline=timeline,T=T, U=U, N_bar=1000, gamma=5, tol=tol, alpha = 0.05)
print(f"after adaptive optimization, the best r_tilde is {r_star}, we obtain the price of {price_est_star}")
#after adaptive optimization, the best r_tilde is 1.5938668642327334, we obtain the price of 0.009531864803386358 and volatility 0.00018868100865645663
#%%
phi_star = (r_star-r)/vol
numerical_prices_star, confidence_intervals_star = generate_numerical_prices(iterations=iterations,Ns=Ns,S_0=S0, K=K, M=M, T=T, timeline=timeline, dt=dt, r=r, vol=vol, generate=generate_stock_path, U=U, phi=phi_star)
numerical_prices_avg_star =  numerical_prices_star.mean()
numerical_prices_std_star = numerical_prices_star.std()
plot_MC_Analytical(analytical_price=Up_Out_price, numerical_prices_avg=numerical_prices_avg_star, numerical_prices_std=numerical_prices_std_star, sample_sizes=Ns, nbr_iterations = nbr_iterations, confidence_level=0.95, filename="../Graphs/MC_UOCO_Optimal_r_tilde", title="Monte Carlo Up and Out call price estimation with optimal IS")
latex_table_star = generate_latex_table(numerical_prices_star, Ns,mean_txt="Mean numerical price", std_txt="estimator's std dev.")
print(latex_table_star)
latex_table_CI_star = generate_latex_table(confidence_intervals_star, Ns, mean_txt="Mean CI size", std_txt="sdt dev. in CI size")
print(latex_table_CI_star)

summary_star_vs_CMC = generate_summary_table(Ns,numerical_prices_std_star,numerical_prices_std, confidence_intervals_star,confidence_intervals, numerical_prices_avg, numerical_prices_avg_star, analytical_price=Up_Out_price)
print(summary_star_vs_CMC)


