import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm

st.set_page_config(page_title="Ch. 6 - FN - TP visualization", layout="wide")

st.title("Chapter 6: Visualizing True Positives and False Negatives in Hypothesis Testing")

# Sidebar user inputs
st.sidebar.header("Simulation Parameters")

# True mean of the population, also used to visualize the alternative hypothesis
mu_true = st.sidebar.slider("True mean (alternative)", -2.0, 2.0, 0.2, 0.01)
# Mean under the null hypothesis
mu_null = st.sidebar.slider("Null hypothesis mean", -2.0, 2.0, 0.0, 0.1)
sigma = st.sidebar.slider("Population standard deviation", 0.1, 5.0, 1.0, 0.01)
n = st.sidebar.slider("Sample size", 5, 1000, 100, 5)
alpha = st.sidebar.slider("Significance level (α)", 0.01, 0.2, 0.05, 0.01)

# Derived parameters
df = n - 1
se = sigma / np.sqrt(n)

# Note: Using the correct t-distributions here, even though the book uses normal distributions for simplicity.
t_crit = t.ppf(1 - alpha / 2, df)

# Null distribution of t-statistic: centered at zero
null_dist = t(df = df, loc=0, scale=1)
# Alternative distribution: t-statistic assuming true mean is mu_true
true_diff = mu_true - mu_null
alt_dist = t(df=df, loc=true_diff / se, scale=1)

# Range of x for plotting
# Compute dynamic x-range based on tails
x_min = min([null_dist.ppf(0.0001), alt_dist.ppf(0.0001)])
x_max = max([alt_dist.ppf(0.9999), null_dist.ppf(0.9999)])
x = np.linspace(x_min, x_max, 1000)

# Two cases: the true mean is either equal or not to the true mean
# If it is not true, recreate dynamically the plot in the book (overlay of two t-stat distributions)
if mu_null != mu_true: 
    fig, ax = plt.subplots()
    ax.plot(x, null_dist.pdf(x), label="If Null were true", color='purple')
    ax.plot(x, alt_dist.pdf(x), label="Alternative is true", color='teal')
    # Fill true positive area (correct rejection)
    x_tp_pos = x[x > t_crit]
    ax.fill_between(x_tp_pos, 0, alt_dist.pdf(x_tp_pos), color='teal', alpha=0.5, label="TP")
    x_tp_neg = x[x < -t_crit]
    ax.fill_between(x_tp_neg, 0, alt_dist.pdf(x_tp_neg), color='teal', alpha=0.5)
    # Fill false negative area (incorrect acceptance)
    x_fn = x[(x > -t_crit) & (x < t_crit)]
    ax.fill_between(x_fn, 0, alt_dist.pdf(x_fn), color='purple', alpha=0.5, label="FN")
    # Critical value lines
    ax.axvline(t_crit, color='black', linestyle='dashed', label="Critical values")
    ax.axvline(-t_crit, color='black', linestyle='dashed')
    # Labels and legend
    ax.set_xlabel("t-statistic")
    ax.set_ylabel("Density")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
    ax.grid(False)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    # Summary stats
    beta = alt_dist.cdf(t_crit) - alt_dist.cdf(-t_crit)
    power = 1 - beta

    # Interpretation
    st.markdown(f"**False Negative Rate (β)**: {beta:.3f}")
    st.markdown(f"**Power (1 - β)**: {power:.3f}")
    st.markdown(
        f"""
        ### How to interpret:
        - The **purple curve** shows the distribution of t-statistics if the null hypothesis is true.
        - The **teal curve** shows the distribution if the alternative hypothesis (true mean) is correct.
        - The **shaded teal area** is where we correctly reject the null (true positive).
        - The **shaded purple area** is where we fail to reject the null even though it's false (false negative).
        - Increasing the **sample size** or the **difference between the mean under the null and the true mean** reduces the purple area (FN) and increases power.
        """
    )

else:
    # If the means are equal, we visualize the true negative and false positive areas
    fig, ax = plt.subplots()
    ax.plot(x, null_dist.pdf(x), label="If Null were true", color='purple')
    ax.plot(x, alt_dist.pdf(x), label="Alternative is true", color='teal')
    # Fill false positive area (inorrect rejection)
    x_fp_pos = x[x > t_crit]
    ax.fill_between(x_fp_pos, 0, alt_dist.pdf(x_fp_pos), color='teal', alpha=0.5, label="FP")
    x_fp_neg = x[x < -t_crit]
    ax.fill_between(x_fp_neg, 0, alt_dist.pdf(x_fp_neg), color='teal', alpha=0.5)
    # Fill true negative area (correct acceptance)
    x_tn = x[(x > -t_crit) & (x < t_crit)]
    ax.fill_between(x_tn, 0, alt_dist.pdf(x_tn), color='purple', alpha=0.5, label="TN")
    # Critical value lines
    ax.axvline(t_crit, color='black', linestyle='dashed', label="Critical values")
    ax.axvline(-t_crit, color='black', linestyle='dashed')
    # Labels and legend
    ax.set_xlabel("t-statistic")
    ax.set_ylabel("Density")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
    ax.grid(False)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    st.pyplot(fig)

    # Interpretation
    st.markdown(
    f"""
    *Note: You have selected the same mean for both the null and alternative hypotheses, which means there is no difference to detect.
    Therefore, the plot instead shows the true negative (TN) and false positive (FP) areas of the decision.*

    ### How to interpret:
    - The **purple curve** shows the distribution of t-statistics if the null hypothesis is true.
    - The **teal curve** shows the distribution if the alternative hypothesis (true mean) is correct (which, in this case, completely overlaps with the other).
    - The **shaded teal area** is where we incorrectly reject the null (false positive). The size of this area directly corresponds to the significance level (α).
    - The **shaded purple area** is where we correctly fail to reject the null (true negative).
    """)
