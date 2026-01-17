import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm

st.set_page_config(page_title="FN - TP visualization", layout="wide")

st.title("Visualizing True Positives and False Negatives in Hypothesis Testing")

# Sidebar user inputs
st.sidebar.header("Simulation Parameters")

# Null hypothesis mean
mu_null = st.sidebar.slider(r"Null hypothesis mean ($\mu_0$)", -2.0, 2.0, 0.0, 0.1,
                            help="This is the value assumed under the null hypothesis (H₀).")

# True mean of the population
mu_true = st.sidebar.slider(r"True population mean ($\mu_{true}$)", -2.0, 2.0, 0.2, 0.1,
                            help=r"This is the actual mean of the population. If this differs from $\mu_0$, the null hypothesis is false.")
# --- FIXED LABELS END ---

sigma = st.sidebar.slider("Standard deviation", 0.1, 5.0, 1.0, 0.1,
                          help="This is the standard deviation of the variable being measured in the population.")
n = st.sidebar.slider("Sample size", 5, 1000, 100, 5,
                      help="This is the number of samples drawn from the population to carry out a single test.")
alpha = st.sidebar.slider(r"Significance level ($\alpha$)", 1, 20, 5, 1,
                          format="%f%%")
alpha = alpha / 100  # Convert to proportion

# Derived parameters
df = n - 1
se = sigma / np.sqrt(n)

t_crit = t.ppf(1 - alpha / 2, df)

# Null distribution of t-statistic: centered at zero
t_null_distribution = t(df = df, loc=0, scale=1)

# Alternative distribution: t-statistic assuming true mean is mu_true
true_diff = mu_true - mu_null
t_alt_distribution = t(df=df, loc=true_diff / se, scale=1)

# Distribution of the actual values
dist_null = norm(loc=mu_null, scale=sigma)
dist_true = norm(loc=mu_true, scale=sigma)

# Range of x for plotting
x_min_t = min([t_null_distribution.ppf(0.0001), t_alt_distribution.ppf(0.0001)])
x_max_t = max([t_alt_distribution.ppf(0.9999), t_null_distribution.ppf(0.9999)])
x_min = min([dist_null.ppf(0.0001), dist_true.ppf(0.0001)])
x_max = max([dist_true.ppf(0.9999), dist_null.ppf(0.9999)])
x = np.linspace(x_min, x_max, 1000)
x_t = np.linspace(x_min_t, x_max_t, 1000)

st.markdown(
    r"""
    This interactive visualization helps illustrate the **concepts of true positives (TP) and false negatives (FN)** in hypothesis testing.
    To do so, we carry out a **two-tailed t-test** comparing a sample to a hypothesized null mean.
    
    Formally, our hypotheses are:
    - **Null hypothesis (H₀):** $\mu = \mu_0$ (The population mean is equal to the benchmark)
    - **Alternative hypothesis (H₁):** $\mu \neq \mu_0$ (The population mean is actually different)

    Usually, we only carry out the test once (as we only have one sample), but this visualization simulates what would happen if we repeated the test many times under the same true population conditions.
    """
)

c1, c2 = st.columns(2)

with c1:
    fig, ax = plt.subplots()
    ax.plot(x, dist_null.pdf(x), label=r"Null Distribution ($\mu_0$)", color='purple')
    ax.plot(x, dist_true.pdf(x), label=r"True Population Distribution ($\mu_{true}$)", color='teal')
    ax.axvline(mu_null, color='black', linestyle='dashed', label="Mean values")
    ax.axvline(mu_true, color='black', linestyle='dashed')
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
    ax.grid(False)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.subheader("Distributions of Variable Values")
    st.caption("The purple curve is what we assume to be true under H₀. The teal curve is the actual reality of the population.")
    st.pyplot(fig)
    st.markdown(
        """
        ### How to interpret:
        - The **purple curve** represents the distribution assumed by the null hypothesis """ + r"($\mu_0 = " + str(mu_null) + r"""$).
        - The **teal curve** represents the actual population distribution """ + r"($\mu_{true} = " + str(mu_true) + r"""$).
        - If these curves overlap significantly, it becomes harder for a test to distinguish between them.
        """)

if mu_null != mu_true: 
    fig, ax = plt.subplots()
    ax.plot(x_t, t_null_distribution.pdf(x_t), label="If H₀ were true", color='purple')
    ax.plot(x_t, t_alt_distribution.pdf(x_t), label="Reality (H₁ is true)", color='teal')
    
    x_tp_pos = x_t[x_t > t_crit]
    ax.fill_between(x_tp_pos, 0, t_alt_distribution.pdf(x_tp_pos), color='teal', alpha=0.5, label="TP")
    x_tp_neg = x_t[x_t < -t_crit]
    ax.fill_between(x_tp_neg, 0, t_alt_distribution.pdf(x_tp_neg), color='teal', alpha=0.5)
    
    x_fn = x_t[(x_t > -t_crit) & (x_t < t_crit)]
    ax.fill_between(x_fn, 0, t_alt_distribution.pdf(x_fn), color='purple', alpha=0.5, label="FN")
    
    ax.axvline(t_crit, color='black', linestyle='dashed', label="Critical values")
    ax.axvline(-t_crit, color='black', linestyle='dashed')
    ax.set_xlabel("t-statistic")
    ax.set_ylabel("Density")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
    ax.grid(False)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    
    with c2:
        st.subheader("Distributions of t-Statistics")
        st.caption("This shows the distribution of t-values under the null and alternative hypotheses. Since μ₀ ≠ μ_true, the null hypothesis is false, so we want to be in the teal (TP) zone.")
        st.pyplot(fig)

        beta = t_alt_distribution.cdf(t_crit) - t_alt_distribution.cdf(-t_crit)
        power = 1 - beta

        st.markdown(
            f"""
            ### How to interpret:
            **False negative rate (β)**: {beta:.3f} — The probability of failing to reject H₀ when it is actually false.
            
            **Power (1 - β)**: {power:.3f} — The probability of correctly rejecting H₀ (true positive rate).

            - **Teal area (TP)**: Correct rejection of a false null.
            - **Purple area (FN)**: Missing a real effect because the t-statistic fell within the critical bounds.
            """
        )

else:
    # Logic for when H0 is actually true (True Negatives and False Positives)
    fig, ax = plt.subplots()
    ax.plot(x_t, t_null_distribution.pdf(x_t), label="Distribution under H₀", color='purple')
    ax.plot(x_t, t_alt_distribution.pdf(x_t), label="Reality (H₀ is true)", color='teal')
    
    x_fp_pos = x_t[x_t > t_crit]
    ax.fill_between(x_fp_pos, 0, t_null_distribution.pdf(x_fp_pos), color='teal', alpha=0.5, label="FP")
    x_fp_neg = x_t[x_t < -t_crit]
    ax.fill_between(x_fp_neg, 0, t_null_distribution.pdf(x_fp_neg), color='teal', alpha=0.5)
    
    x_tn = x_t[(x_t > -t_crit) & (x_t < t_crit)]
    ax.fill_between(x_tn, 0, t_null_distribution.pdf(x_tn), color='purple', alpha=0.5, label="TN")
    
    ax.axvline(t_crit, color='black', linestyle='dashed', label="Critical values")
    ax.axvline(-t_crit, color='black', linestyle='dashed')
    ax.set_xlabel("t-statistic")
    ax.set_ylabel("Density")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
    ax.grid(False)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    
    with c2:
        st.subheader("Distributions of t-Statistics")
        st.caption(r"This shows the distribution of t-values under the null and alternative hypotheses. Based on the provided parameters, the null hypothesis is true ($\mu_0 = \mu_{true}$).")
        st.pyplot(fig)
        st.markdown(f"""
        ### How to interpret:
        - **Shaded teal (false positive)**: We rejected H₀ even though it was true (Type I error). This equals your α ({alpha}).
        - **Shaded purple (true negative)**: We correctly failed to reject the null hypothesis.
        """)