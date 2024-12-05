import math
import numpy as np
import scipy.stats as st

def analytical_call_price(S_0, K, T, r, sigma):
    d1 = (np.log(S_0/K)+(r+(sigma**2)/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S_0*st.norm.cdf(d1) - np.exp(-r*T)*K*st.norm.cdf(d2)

def generate_latex_table(values, sample_sizes, mean_txt, std_txt):
    # Extract means and stds for the values
    means = values.mean()
    stds = values.std()

    # Start LaTeX table construction
    latex_table = r"""
\begin{table}[h!]
\centering
\caption{Confidence Intervals for Different Sample Sizes}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{l""" + "c" * len(sample_sizes) + r"""}
\toprule
\textbf{Sample size} & """ + " & ".join([f"$10^{{{int(math.log10(size))}}}$" for size in sample_sizes]) + r""" \\
\midrule
""" + mean_txt + r""" & """ + " & ".join([f"{mean:.5f}" for mean in means]) + r""" \\
""" + std_txt + r""" & """ + " & ".join([f"{std:.5f}" for std in stds]) + r""" \\
\bottomrule
\end{tabular}
\end{adjustbox}
\label{tab:confidence_intervals}
\end{table}
"""
    return latex_table

def generate_summary_table(sample_sizes,numerical_prices_std_IS,numerical_prices_std, confidence_intervals_IS,confidence_intervals, numerical_prices_avg, numerical_prices_avg_IS, analytical_price):

    variance_reduction_ratio = (numerical_prices_std_IS / numerical_prices_std) ** 2
    relative_std_change = (numerical_prices_std - numerical_prices_std_IS) / numerical_prices_std
    relative_var_change = (numerical_prices_std ** 2 - numerical_prices_std_IS ** 2) / numerical_prices_std ** 2
    relative_change_CI = np.abs(
        confidence_intervals_IS.mean() - confidence_intervals.mean()) / confidence_intervals.mean()
    absolute_error = np.abs(numerical_prices_avg - analytical_price)
    absolute_error_IS = np.abs(numerical_prices_avg_IS - analytical_price)
    absolute_error_ratio = absolute_error_IS / absolute_error
    # Start LaTeX table construction
    latex_table = r"""
\begin{table}[h!]
\centering
\caption{Summary of Metrics for Different Sample Sizes}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{l""" + "c" * len(sample_sizes) + r"""}
\toprule
\textbf{Metric} & """ + " & ".join([f"$10^{{{int(len(str(size)) - 1)}}}$" for size in sample_sizes]) + r""" \\
\midrule
Variance Reduction Ratio & """ + " & ".join([f"{val:.4f}" for val in variance_reduction_ratio]) + r""" \\
Relative Std Change & """ + " & ".join([f"{val:.4f}" for val in relative_std_change]) + r""" \\
Relative Variance Change & """ + " & ".join([f"{val:.4f}" for val in relative_var_change]) + r""" \\
Relative Change CI & """ + " & ".join([f"{val:.4f}" for val in relative_change_CI]) + r""" \\
Absolute Error & """ + " & ".join([f"{val:.4f}" for val in absolute_error]) + r""" \\
Absolute Error (IS) & """ + " & ".join([f"{val:.4f}" for val in absolute_error_IS]) + r""" \\
Absolute Error Ratio & """ + " & ".join([f"{val:.4f}" for val in absolute_error_ratio]) + r""" \\
\bottomrule
\end{tabular}
\end{adjustbox}
\label{tab:summary_metrics}
\end{table}
"""
    return latex_table


