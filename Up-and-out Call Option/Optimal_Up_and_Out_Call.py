
import numpy as np
import polars as pl
from Algorithms.crank_nicholson import crank_nicholson_pde_solver, zeta_control_generator
from Algorithms.generation import generate_controled_path, generate_numerical_prices
from Utils.grapher import plot_paths, surface_plotter, plot_MC_Analytical
from scipy.interpolate import RegularGridInterpolator
from Utils.helpers import analytical_Up_Out_call_price

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
#Subintervals
M = 1000

# We will assume t = 0 for simplicity
dt = T/M
timeline = np.linspace(0,T,num=M+1,endpoint=True)

#Grid building
P = 3000 #space
#Subintervals
M_tilde = M #time
s_min = S0*np.exp((r-(vol**2)/2)*T-6*vol*np.sqrt(T))
s_max = U

s_set = np.linspace(s_min,s_max,num=P,endpoint=True)
t_set = np.linspace(0,T,num=M_tilde,endpoint=True)

dt = T/(M_tilde-1)
ds = (s_max-s_min)/(P-1)

Up_Out_price = analytical_Up_Out_call_price(S0, K, U, T, r, vol)
print(Up_Out_price)
#%%

#Boundary conditions
eps = 0.01

#Setup value function with its boundary condition
#V(s,t), value function
V_matrix = np.zeros((P,M_tilde))
#condition for v(smin,t) already 0
#condition for v(smax,t)
V_matrix[-1,:] = eps
#condition for v(s,T)
V_matrix[:,-1] = np.maximum(s_set-K,0)


V_matrix = crank_nicholson_pde_solver(V_matrix, dt, ds, s_set, M_tilde, P, vol, r)

#surface_plotter(s_set, t_set,V_matrix, title="Value Surface", filename="V_surface_UOC.png")

#%%
zeta = zeta_control_generator(P,M_tilde, V_matrix, s_set, ds, vol)

S_min = S0*np.exp((r-(vol**2)/2)*T-3*vol*np.sqrt(T))
S_max = S0*np.exp((r-(vol**2)/2)*T+3*vol*np.sqrt(T))

idx_S_min = np.argmin(np.abs(s_set - S_min))
idx_S_max = np.argmin(np.abs(s_set - S_max))

surface_plotter(s_set[idx_S_min:idx_S_max], t_set[:-1],zeta[idx_S_min:idx_S_max,:-1], title="Zeta Optimal Control Surface", filename="Graphs/S_surface_UOC.png")


zeta_interpolator = RegularGridInterpolator((s_set, t_set), zeta, method='linear', bounds_error=False, fill_value=0)

#%%
stock_price, _ = generate_controled_path(S_0=S0, nbr_gen=100, time_obs=M_tilde, timeline=t_set, delta_t=dt,mu=r,sigma=vol, s_min=s_min, s_max=s_max, zeta_interpolator=zeta_interpolator)


stock_price = stock_price.with_columns(
                (np.maximum(pl.col(f"S_{t_set[-1]}") - K, 0)* np.exp(-r * T)).alias("payoff")
            )


stock_price = stock_price.with_columns(
    pl.max_horizontal(pl.exclude("likelihood_ratio")).alias("S_max")
)
stock_price = stock_price.with_columns(
    (pl.when(pl.col("S_max") <= U).then(1).otherwise(0)).alias("condition")
)

stock_price = stock_price.with_columns(
    (pl.col("payoff") * pl.col("likelihood_ratio") * pl.col("condition")).alias("payoff_weighted")
)

value = (stock_price.select("payoff_weighted")).mean().item()

print("The price is:",value)
N =100
plot_paths(stock_paths=stock_price.select(pl.exclude(["likelihood_ratio", "payoff", "payoff_weighted", "S_max", "condition"])).to_numpy(), timeline=t_set, N=N, K=K, filename=f"Graphs/stock_paths_optimal_{N}_{P}s_{M_tilde}t_UOC.png", avg_path=True, U=U, title="Generated Controlled Stock Price Paths with 30% volatility")

#%%
#Smapling size
nbr_Ns = 5
Ns = np.logspace(1,4,num=nbr_Ns,dtype=int) #np.ndarray

# iteration of each process
nbr_iterations = 7
iterations = np.arange(0,nbr_iterations) #np.ndarray

numerical_prices_opt_control, confidence_intervals_opt_control = generate_numerical_prices(iterations=iterations,Ns=Ns,S_0=S0, K=K, M=M_tilde, T=T, timeline=t_set, dt=dt, r=r, vol=vol, generate=generate_controled_path, U = U, s_min=s_min, s_max=s_max, zeta_interpolator=zeta_interpolator)
numerical_prices_avg_opt_control=  numerical_prices_opt_control.mean()
numerical_prices_std_opt_control = numerical_prices_opt_control.std()
plot_MC_Analytical(analytical_price=Up_Out_price, numerical_prices_avg=numerical_prices_avg_opt_control, numerical_prices_std=numerical_prices_std_opt_control, sample_sizes=Ns, nbr_iterations = nbr_iterations, confidence_level=0.95, filename="Graphs/MC_UOCO_Optimal_Controlled")

