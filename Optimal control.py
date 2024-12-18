import numpy as np
from scipy.linalg import solve_banded, solve
import matplotlib.pyplot as plt
import plotly.graph_objects as go

#Interest rate
r = .05
#stock volatility
vol = 0.1

#Option maturity
T = 0.2
#initial asset price
S0 = 100
#strike price
K = 120
#Subintervals
M = 100

#%%
#Grid building
P = 50
M_tilde = 30
s_min = S0*np.exp((r-(vol**2)/2)*T-6*vol*np.sqrt(T))
s_max = S0*np.exp((r-(vol**2)/2)*T+6*vol*np.sqrt(T))

s_set = np.linspace(s_min,s_max,num=P,endpoint=True)
t_set = np.linspace(0,T,num=M_tilde,endpoint=True)

dt = T/(M_tilde-1)
ds = (s_max-s_min)/(P-1)

#Setup value function with its boundary condition
#V(s,t), value function
V_matrix = np.zeros((P,M_tilde))
#condition for v(s,T)
V_matrix[:,-1] = np.maximum(s_set-K,0)
#condition for v(smin,t) already 0
#condition for v(smax,t)
V_matrix[-1,:] = s_max-K*np.exp(-r*(T-t_set))

#%%
#As descibed in the report build the M1 matrix
d_m = np.array([1/dt + 2*((s*vol)**2)/(4*(ds**2)) for s in s_set[1:-1]])
u_m = np.array([-(r*s)/(4*ds)-(s*vol)**2/(4*(ds**2)) for s in s_set[1:-2]])
l_m = np.array([(r*s)/(4*ds)-(s*vol)**2/(4*(ds**2)) for s in s_set[2:-1]])

# Constructing the tridiagonal matrix
#M1_matrix = np.diag(d_m, k=0) + np.diag(u_m, k=1) + np.diag(l_m, k=-1)

M_m = np.zeros((3, P-2))
M_m[0, 1:] = u_m
M_m[1, :] = d_m
M_m[2, :-1] = l_m
#M_m = np.diag(d_m, k=0) + np.diag(u_m, k=1) + np.diag(l_m, k=-1)

#As descibed in the report build the M2 matrix
d_mp1 = np.array([1/dt - 2*((s*vol)**2)/(4*(ds**2)) for s in s_set[1:-1]])
u_mp1 = np.array([(r*s)/(4*ds)+(s*vol)**2/(4*(ds**2)) for s in s_set[1:-2]])
l_mp1 = np.array([-(r*s)/(4*ds)+(s*vol)**2/(4*(ds**2)) for s in s_set[2:-1]])

M_mp1 = np.diag(d_mp1, k=0) + np.diag(u_mp1, k=1) + np.diag(l_mp1, k=-1)
#%%
#Iterate backward in time
for j in range(M_tilde - 2, -1, -1):

    b_m = np.zeros(P-2)
    b_m[0] = 0
    b_m[-1] = (-(r*s_set[-1])/(4*ds)-(s_set[-1]*vol)**2/(4*ds**2))*V_matrix[-1, j]

    b_mp1 = np.zeros(P-2)
    b_mp1[0] = 0
    b_mp1[-1] = ((r*s_set[-1])/(4*ds)-(s_set[-1]*vol)**2/(4*ds**2)) * V_matrix[-1, j+1]

    y = np.dot(M_mp1, V_matrix[1:-1, j + 1])+b_mp1-b_m

    # Solve the banded system
    V_matrix[1:-1, j] = solve_banded((1, 1), M_m, y)
    #V_matrix[1:-1, j] = solve(M_m,y)

#Plot the 3D shape of the value over time and interest rate
S, T = np.meshgrid(s_set, t_set)

# Create the interactive 3D plot
fig = go.Figure()


# Add a surface trace
fig.add_trace(go.Surface(z=V_matrix.T, x=S, y=T, colorscale='Viridis'))

# Update layout for labels and title
fig.update_layout(
    scene=dict(
        xaxis_title=r'Asset Value $(s_t)$',
        yaxis_title='Time to Maturity (t)',
        zaxis_title='Put Option Price',
    ),
    title='Interactive 3D Plot of Put Option Price',
)
fig.write_image("V_surface.png",width=1920, height=1080, scale=1)
fig.show()

#%%
camera = dict(
    eye=dict(x=-2, y=-10, z=1)  # Adjust x, y, z to rotate the plot
)

zeta_forward = np.zeros((P,M_tilde))
for j in range(M_tilde):
    forward_diff = np.diff(V_matrix[:, j], prepend=V_matrix[1, j])/ds
    #print(forward_diff)

    zeta_forward[:,j] = forward_diff*vol*(s_set/V_matrix[:,j])

#Plot the 3D shape of the value over time and interest rate
S, T = np.meshgrid(s_set[3:-3], t_set[:-3])

# Create the interactive 3D plot
fig = go.Figure()

# Add a surface trace
fig.add_trace(go.Surface(z=zeta_forward[3:-3,:-3].T, x=S, y=T, colorscale='Viridis'))

# Update layout for labels and title
fig.update_layout(
    scene=dict(
        xaxis_title=r'Asset Value $(s_t)$',
        yaxis_title='Time to Maturity (t)',
        zaxis_title='Put Option Price',
    ),
    title='Interactive 3D Plot of Put Option Price',
)
fig.write_image("Z_surface.png",width=1920, height=1080, scale=1)
fig.show()

import polars as pl
import scipy.stats as st
def generate_controled_path(S_0, nbr_gen, time_obs, timeline, delta_t,mu,sigma, zeta, phi = 0):
    stock_price = pl.DataFrame({"S_0.0": pl.Series([S_0] * nbr_gen)})
    for t in range(time_obs):
        #brownian increment, same as Wt+1-Wt
        #TODO
        # define the s in the zeta, perhaps find the closest match for control
        dW = st.norm.rvs(loc = zeta[s,t], scale=np.sqrt(delta_t), size = nbr_gen)

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



