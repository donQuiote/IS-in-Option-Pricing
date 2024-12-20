N_bar = 100
gamma = 2
tol = 1.5E-3
alpha = 0.05
c = st.norm.ppf(1-alpha/2)
vol_Z = np.inf
N = N_bar/gamma
r_tilde = (K/S0-1)/T
phi = (r_tilde-r)/vol
mu_Z = 0

def adaptive_optimal_r():

    while (vol_Z*c)/np.sqrt(N)>tol and N<20E6:
        N=int(gamma*N)
        stock_price = pl.DataFrame({"S_0.0": pl.Series([S0] * N)})
        dW_sum = np.zeros(N)
        for t in range(M):
            # brownian increment, same as Wt+1-Wt
            dW = st.norm.rvs(loc=phi * dt, scale=np.sqrt(dt), size=N)

            # add observations
            stock_price = stock_price.with_columns(
                (pl.col(f"S_{timeline[t]}") * (1 + r * dt + vol * dW)).alias(f"S_{timeline[t + 1]}")
            )

            stock_prices = stock_price.with_columns(
                pl.Series(f"LR_{timeline[t+1]}", 0.5 * dt * phi ** 2 - phi * dW)
            )

            # If phi is zero this won't do anything
            if phi != 0:
                dW_sum += dW


        stock_price = stock_price.with_columns(
            pl.Series("likelihood_ratio", np.exp(0.5*dt*M*phi**2-phi*dW_sum))
        )

        final_stock = f"S_{T}"

        stock_price = stock_price.with_columns(
            (np.maximum(pl.col(final_stock) - K, 0)* np.exp(-r * T)).alias("payoff")
        )

        stock_price = stock_price.with_columns(
            (pl.col("payoff") * pl.col("likelihood_ratio")).alias("payoff_weighted")
        )

        mu_Z = (stock_price.select("payoff_weighted")).mean().item()

        vol_Z =  (stock_price.select("payoff_weighted")).std(ddof=1).item()

        def phi_star_finder(phi_star, stock_price):
            stock_price = stock_price.with_columns(
                pl.Series("correction", np.exp(0.5 * dt * M * (phi ** 2+phi_star**2) - (phi+phi_star) * dW_sum))
            )
            stock_price = stock_price.with_columns(
                (pl.col("payoff").pow(2) * pl.col("likelihood_ratio")*pl.col("correction")).alias("to_minimize")
            )
            value = (stock_price.select("to_minimize")).mean().item()
            return value

        phi = minimize(phi_star_finder, x0=1.2, args=(stock_price,), tol=1e-6).x.item()

        print(f"the optimal r tilde is {phi*vol+r}")
        print(f"{(vol_Z*c)/np.sqrt(N)},{N}")

    #def phi_minimizer(r_hat):
print(f"after adaptive optimization, the best r_tilde is {phi*vol+r}, we obtain the price of {mu_Z} and volatility {vol_Z}")
print(analytical_call_price(S_0=S0,K=K,T=T,r=r,sigma=vol))




#Plot the 3D shape of the value over time and interest rate
S_mesh, T_mesh = np.meshgrid(s_set[idx_S_min:idx_S_max], t_set)

# Create the interactive 3D plot
fig = go.Figure()

# Add a surface trace
fig.add_trace(go.Surface(z=zeta_forward[idx_S_min:idx_S_max].T, x=S_mesh, y=T_mesh, colorscale='Viridis'))

# Update layout for labels and title
fig.update_layout(
    scene=dict(
        xaxis_title=r'Asset Value $(s_t)$',
        yaxis_title='Time to Maturity (t)',
        zaxis_title='Put Option Price',
    ),
    title='Zeta optimal control',
)
fig.write_image("Z_surface.png",width=1920, height=1080, scale=1)
fig.show()