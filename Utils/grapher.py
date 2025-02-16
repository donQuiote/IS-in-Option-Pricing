import matplotlib.pyplot as plt
from scipy.stats import t
import numpy as np
import plotly.graph_objects as go


def plot_paths(stock_paths,timeline,N,K,filename, avg_path= False, U=np.inf, title = "Generated Stock Price Paths with 51% volatility"):
    plt.figure(num=2, figsize=(16, 8), dpi=600)
    for i in range(N):
        plt.plot(timeline, stock_paths[i, :])

    if avg_path:
        avg_path = np.mean(stock_paths, axis=0)
        plt.plot(timeline, avg_path, color="red", linewidth=2, label="Average Path")
    if U != np.inf:
        plt.hlines(U, xmin=timeline[0], xmax=timeline[-1], colors="red", linestyles="dotted", linewidth=1.5, label=f"Barrier {U}")
    plt.hlines(K, xmin=timeline[0], xmax=timeline[-1], colors="black", linestyles="dotted", linewidth=1.5, label=f"strike {K}")
    plt.xlabel("Time", fontsize=16)
    plt.ylabel("Stock Price", fontsize=16)
    plt.title(title, fontsize=20)
    plt.legend(fontsize=16)

    # Enhance tick labels
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_MC_Analytical(analytical_price, numerical_prices_avg, numerical_prices_std, sample_sizes,nbr_iterations, confidence_level,filename, title = "Crude Monte Carlo call price estimation in function of sample size"):
    plt.figure(num = 1,figsize=(16,8),dpi=200)
    plt.hlines(analytical_price,xmin=sample_sizes[0],xmax=sample_sizes[-1],colors="red", linestyles="dotted", label=f"analytical price")
    plt.plot(sample_sizes,numerical_prices_avg, color='blue')

    # CI using students distrib
    dof = nbr_iterations - 1
    t_critical = t.ppf(1 - (1 - confidence_level) / 2, dof)
    error = t_critical * (numerical_prices_std / np.sqrt(sample_sizes))
    lower_bound = numerical_prices_avg - error
    upper_bound = numerical_prices_avg + error
    plt.fill_between(sample_sizes, lower_bound, upper_bound, color='blue', alpha=0.2, label=f"{int(confidence_level * 100)}% Confidence Interval")

    plt.xscale('log')
    plt.xlabel("sample sizes [N]", fontsize=16)
    plt.ylabel("call option price with", fontsize=16)
    plt.title(title, fontsize=20)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Enhance tick labels
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.savefig(filename, bbox_inches='tight')

    plt.show()

def surface_plotter(s_set, t_set,V_matrix, title, filename):
    #Plot the 3D shape of the value over time and interest rate
    S_mesh, T_mesh = np.meshgrid(s_set, t_set)

    # Create the interactive 3D plot
    fig = go.Figure()


    # Add a surface trace
    fig.add_trace(go.Surface(z=V_matrix.T, x=S_mesh, y=T_mesh, colorscale='Viridis'))

    # Update layout for labels and title
    fig.update_layout(
        scene=dict(
            xaxis_title=r"Asset Value S",
            yaxis_title="Time to Maturity (t)",
            zaxis_title="Value",
        ),
        title=title,
    )
    fig.write_image(filename,width=1920, height=1080, scale=1)
    fig.show()
