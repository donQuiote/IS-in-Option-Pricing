#%%
import numpy as np
import polars as pl

from Algorithms.generation import generate_controled_path
from Utils.grapher import plot_paths, surface_plotter
from Algorithms.crank_nicholson import crank_nicholson_pde_solver, zeta_control_generator
from scipy.interpolate import RegularGridInterpolator
from Algorithms.generation import generate_numerical_prices
from Utils.helpers import analytical_call_price
from Utils.grapher import plot_MC_Analytical

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

#grid_params = [(50,30),(100,60),(300,180),(500,300)]
grid_params = [(300,180)]

for grid_param in grid_params:

    #Grid building
    P = grid_param[0] #space
    #Subintervals
    M_tilde = grid_param[1] #time

    s_min = S0*np.exp((r-(vol**2)/2)*T-6*vol*np.sqrt(T))
    s_max = S0*np.exp((r-(vol**2)/2)*T+6*vol*np.sqrt(T))

    s_set = np.linspace(s_min,s_max,num=P,endpoint=True)
    t_set = np.linspace(0,T,num=M_tilde,endpoint=True)

    dt = T/(M_tilde-1)
    ds = (s_max-s_min)/(P-1)

    #%%

    #Setup value function with its boundary condition
    #V(s,t), value function
    V_matrix = np.zeros((P,M_tilde))
    #condition for v(s,T)
    V_matrix[:,-1] = np.maximum(s_set-K,0)
    #condition for v(smin,t) already 0
    #condition for v(smax,t)
    V_matrix[-1,:] = s_max-K*np.exp(-r*(T-t_set))

    V_matrix = crank_nicholson_pde_solver(V_matrix, dt, ds, s_set, M_tilde, P, vol, r)

    surface_plotter(s_set, t_set, V_matrix, title="Value Surface", filename="../Graphs/V_surface.png")

    #%%
    zeta = zeta_control_generator(P,M_tilde, V_matrix, s_set, ds, vol)

    S_min = S0*np.exp((r-(vol**2)/2)*T-3*vol*np.sqrt(T))
    S_max = S0*np.exp((r-(vol**2)/2)*T+3*vol*np.sqrt(T))

    idx_S_min = np.argmin(np.abs(s_set - S_min))
    idx_S_max = np.argmin(np.abs(s_set - S_max))

    surface_plotter(s_set[idx_S_min:idx_S_max], t_set[:-1],zeta[idx_S_min:idx_S_max,:-1], title="Zeta Optimal Control Surface", filename="Graphs/S_surface.png")

    zeta_interpolator = RegularGridInterpolator((s_set, t_set), zeta, method='linear', bounds_error=False, fill_value=0)
    #%%
    N = 100
    stock_price, _ = generate_controled_path(S_0=S0, nbr_gen=N, time_obs=M_tilde, timeline=t_set, delta_t=dt,mu=r,sigma=vol, s_min=s_min, s_max=s_max, zeta_interpolator=zeta_interpolator)


    stock_price = stock_price.with_columns(
                    (np.maximum(pl.col(f"S_{t_set[-1]}") - K, 0)* np.exp(-r * T)).alias("payoff")
                )

    stock_price = stock_price.with_columns(
        (pl.col("payoff") * pl.col("likelihood_ratio")).alias("payoff_weighted")
    )

    value = (stock_price.select("payoff_weighted")).mean().item()

    print("The price is:",value)

    plot_paths(stock_paths=stock_price.select(pl.exclude(["likelihood_ratio", "payoff", "payoff_weighted"])).to_numpy(), timeline=t_set, N=N, K=K, filename=f"Graphs/stock_paths_optimal_{N}_{P}s_{M_tilde}t.png", avg_path=True, title="Generated Controlled Stock Price Paths with 51% volatility")

    #Smapling size
    nbr_Ns = 10
    Ns = np.logspace(1,4,num=nbr_Ns,dtype=int) #np.ndarray

    # iteration of each process
    nbr_iterations = 12
    iterations = np.arange(0,nbr_iterations) #np.ndarray

    numerical_prices_opt_control, confidence_intervals_opt_control = generate_numerical_prices(iterations=iterations,Ns=Ns,S_0=S0, K=K, M=M_tilde, T=T, timeline=t_set, dt=dt, r=r, vol=vol,  generate=generate_controled_path, s_min=s_min, s_max=s_max, zeta_interpolator=zeta_interpolator)
    numerical_prices_avg_opt_control=  numerical_prices_opt_control.mean()
    numerical_prices_std_opt_control = numerical_prices_opt_control.std()
    analytical_price = analytical_call_price(S_0=S0,K=K,T=T,r=r,sigma=vol)
    plot_MC_Analytical(analytical_price=analytical_price, numerical_prices_avg=numerical_prices_avg_opt_control, numerical_prices_std=numerical_prices_std_opt_control, sample_sizes=Ns, nbr_iterations = nbr_iterations, confidence_level=0.95, filename=f"Graphs/MC_CO_Optimal_Controlled_{P}s_{M_tilde}t", title="Monte Carlo call price estimation with optimal control")
    print("average price",numerical_prices_avg_opt_control)
    print("average std",numerical_prices_std_opt_control)