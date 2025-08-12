import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer
import statsmodels.formula.api as smf

import warnings
warnings.filterwarnings("ignore")

color = ["#3a5e8c", "#10a53d", "#541352", "#ffcf20", "#2f9aa0"]

st.set_page_config(page_title="Ch. 21 - Common Support", layout="wide")

st.title("Ch. 21 - Common Support")

st.markdown("""
### Introduction

This dashboard illustrates the concept of **common support** in causal inference and demonstrates how it affects estimation of treatment effects. 

### Key Concepts

- **Treatment & Outcome**:  
  We model the effect of having a college degree (`has_degree`) on log annual income (`log_income`), controlling for family background measured by log parental income (`log_parent_income`).

- **Common Support**:  
  Common support refers to the overlapping range of covariate values (here, `log_parent_income`) where both treated (`has_degree = 1`) and control (`has_degree = 0`) units exist. Reliable causal effect estimation requires restricting analysis to this region to avoid extrapolation.

- **Coarsened Exact Matching (CEM)**:  
  CEM groups data into bins of `log_parent_income` and compares treated and control units within these bins. Only bins containing both treated and control observations contribute to the treatment effect estimate.

- **Ordinary Least Squares (OLS)**:  
  Two OLS estimates are provided:  
  - **Naive OLS** on the full sample ignores common support and may be biased.  
  - **OLS on Common Support** restricts analysis to the overlapping range of `log_parent_income`, improving estimation validity.

### Functionalities

- **Data Generation Controls**  
  Customize the sample size, complexity of the data generating process (e.g., polynomial or interaction effects), and noise level to explore different scenarios.

- **Binning Settings**  
  Adjust the number of bins and binning strategy for CEM to observe how matching quality affects estimates.

- **Visualizations**  
  - Scatterplot and distribution of `log_income` vs `log_parent_income` by treatment status.  
  - Summary statistics of the generated data.  
  - Bar plot comparing true treatment effect with estimates from different methods.

Use the sidebar controls to experiment with the data and see how restricting to the common support region changes causal effect estimates.
""")
# Sidebar for settings
st.sidebar.header("Settings")
# data generation parameters
st.sidebar.subheader("Data Generation Parameters")
n_samples = st.sidebar.slider("Sample Size", 1000, 5000, 1000)
log_pincome_meandiff = st.sidebar.slider("Log Parental Income Mean Difference", 0.0, 2.0, 1.25)
funcform = st.sidebar.selectbox("Add complexity to data generating process", ["None", "Polynomial", "Interaction"], index=0)
if funcform == 'Polynomial':
    log_pincome_degree = st.sidebar.slider("Log Parental Income Polynomial Degree", min_value=2, max_value=4, value=2)
    log_pincome_coeffs = []
    for d in range(1, log_pincome_degree + 1):
        log_pincome_coeffs.append(st.sidebar.number_input(f"Log Parental Income Coefficient (Degree {d})", value=0.0))
elif funcform == 'Interaction':
    log_pincome_degree = 1
    log_pincome_coeffs = []
    log_pincome_coeffs.append(st.sidebar.number_input("Log Parental Income Coefficient", value=0.0))
    interaction_coef = st.sidebar.number_input("Interaction Coefficient", value=0.0)
else:
    log_pincome_degree = 1
    log_pincome_coeffs = []
    log_pincome_coeffs.append(st.sidebar.number_input("Log Parental Income Coefficient", value=0.0))

noise_std = st.sidebar.slider("Standard Deviation of Noise", 0.0, 1.0, 0.0)
seed = st.sidebar.number_input("Random Seed", value=42, min_value=0, step=1)
np.random.seed(seed)
# binning parameters
st.sidebar.subheader("Binning Parameters")
nbins_parent_income = st.sidebar.slider("Number of Bins for Log Parental Income", 5, 500, 100)
binning_strat = st.sidebar.selectbox("Binning Strategy",["Equal width", "Equal frequency"],index=1)
binning_strat = 'uniform' if binning_strat == "Equal width" else 'quantile'

# Generate synthetic data, may need to tweak parameters later to be more realistic
def generate_data(n_samples, log_pincome_meandiff, noise_std):
    degree = np.random.binomial(1, 0.5, n_samples)
    z = np.random.normal(loc=0, scale=1, size=n_samples)
    log_pincome = z * 0.5 + 11 + degree * log_pincome_meandiff
    log_pincome = np.clip(log_pincome, 9, 14)

    log_income = (
        2
        + 0.2 * degree
        + sum(c * (log_pincome ** (d+1)) for d, c in enumerate(log_pincome_coeffs))
        + (interaction_coef * log_pincome * degree if funcform == 'Interaction' else 0)
        + np.random.normal(0, noise_std, n_samples)
    )

    log_income_nodegree = (
        2
        + 0.2 * 0
        + sum(c * (log_pincome ** (d+1)) for d, c in enumerate(log_pincome_coeffs))
        + (interaction_coef * log_pincome * 0 if funcform == 'Interaction' else 0)
    )

    log_income_degree = (
        2
        + 0.2 * 1
        + sum(c * (log_pincome ** (d+1)) for d, c in enumerate(log_pincome_coeffs))
        + (interaction_coef * log_pincome * 1 if funcform == 'Interaction' else 0)
    )

    data = pd.DataFrame({
        'log_income': log_income,
        'has_degree': degree,
        'log_parent_income': log_pincome
    })

    effects = log_income_degree - log_income_nodegree
    return data, effects

# Calculate effect estimate using coarsaned exact matching
def cem_coef(data):
    # Discretize log parental income into bins
    discretizer = KBinsDiscretizer(
        n_bins=nbins_parent_income, 
        encode='ordinal', 
        strategy=binning_strat
    )
    data['parent_income_bin'] = discretizer.fit_transform(
        data[['log_parent_income']]
    ).astype(int)

    # Group by treatment status and bins
    treated = (
        data[data['has_degree'] == 1]
        .groupby('parent_income_bin')
        .agg(mean_log_income=('log_income', 'mean'),
             count=('log_income', 'count'))
        .rename(columns={'mean_log_income': 'treated_mean_log_income',
                         'count': 'treated_count'})
    )

    control = (
        data[data['has_degree'] == 0]
        .groupby('parent_income_bin')
        .agg(mean_log_income=('log_income', 'mean'),
             count=('log_income', 'count'))
        .rename(columns={'mean_log_income': 'control_mean_log_income',
                         'count': 'control_count'})
    )

    # Match bins with both treated and control observations
    matched = treated.join(control, how='inner')
    matched['total_count'] = matched['treated_count'] + matched['control_count']
    matched['treatment_effect'] = (
        matched['treated_mean_log_income'] - matched['control_mean_log_income']
    )

    # Weighted average treatment effect
    ate = np.average(
        matched['treatment_effect'], 
        weights=matched['total_count']
    ) if not matched.empty else np.nan

    return len(matched), ate


def ols_coef(data):
    model = smf.ols('log_income ~ has_degree + log_parent_income', data=data).fit()
    coef = model.params['has_degree']
    return coef

# calculate OLS effect estimate on the common support region
def ols_coef_with_support(data):

    income_min = max(data[data.has_degree == 0].log_parent_income.min(), data[data.has_degree == 1].log_parent_income.min())
    income_max = min(data[data.has_degree == 0].log_parent_income.max(), data[data.has_degree == 1].log_parent_income.max())

    mask = (
        (data["log_parent_income"] >= income_min) & (data["log_parent_income"] <= income_max)
    )

    data_matched = data[mask]

    model = smf.ols('log_income ~ has_degree + log_parent_income', data=data_matched).fit()
    coef = model.params['has_degree']
    return len(data_matched), coef

data, effects = generate_data(n_samples, log_pincome_meandiff, noise_std)

# data generating process
st.subheader("Data Generating Process")
formula = r'$ln(income) = 2 + 0.2 * has\_degree + ' + str(round(log_pincome_coeffs[0],2)) + r' * ln(parent\_income)'
if funcform == 'Polynomial':
    for d, c in enumerate(log_pincome_coeffs[1:], start=2):
        formula += r' + ' + str(round(c,2)) +  r' * ln(parent\_income)^' + str(d)
elif funcform == 'Interaction':
    formula += r' + ' + str(round(interaction_coef,2)) + r' * ln(parent\_income) * has\_degree'
formula += r' + \epsilon$'

st.markdown(formula)

# summary statistics
st.subheader("Data Summary")
summary_frame = data.describe().T
summary_frame.index = ['Log Income', 'Has Degree', 'Log Parental Income']
st.dataframe(summary_frame)

data['Has degree'] = data['has_degree'].map({0: "No", 1: "Yes"})

# Scatterplots of generated data
st.subheader("Scatterplot and Conditional Distribution")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Log Income vs Log Parental Income")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=data['log_parent_income'], y=data['log_income'], hue=data['Has degree'],
                    alpha=0.6, ax=ax, palette=color)
    ax.set_xlabel("Log Parental Income")
    ax.set_ylabel("Log Annual Income")
    st.pyplot(fig)
with col2:
    st.subheader("Distribution of Log Parental Income by Degree Status")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=data, x='log_parent_income', hue='Has degree', fill=True, ax=ax, palette=color)
    ax.set_xlabel("Log Parental Income")
    st.pyplot(fig)


# calculate estimates
bins_with_support, cem_estimate = cem_coef(data)
ols_full = ols_coef(data)
obs_with_support, ols_cs = ols_coef_with_support(data)
true_ate = np.mean(effects)

st.subheader("Effect Estimation Summary")

col1, col2 = st.columns(2)
# display written summary of results
with col1:
    st.markdown(f"""
    Out of {nbins_parent_income} possible bins, {bins_with_support} bins had matches for coarsaned matching.

    Out of {n_samples} total observations, {obs_with_support} observations fall within the common support region of the distributions.

    The estimates of the treatment effect are as follows:

    - **True Effect**: {true_ate:.3f}
    - **Naive OLS (Full Sample)**: {ols_full:.3f}
    - **OLS (Common Support)**: {ols_cs:.3f}
    - **Coarsened Exact Matching**: {cem_estimate:.3f}
    """)

result_df = pd.DataFrame({
    "Estimator": [
        "Naive OLS\n(Full Sample)",
        "OLS\n(Common Support)",
        "Coarsened\nExact\nMatching"
    ],
    "Estimate": [ols_full, ols_cs, cem_estimate]
})

# Bar plot of estimates
with col2:
    fig, ax = plt.subplots()
    sns.barplot(data=result_df, x="Estimator", y="Estimate", ax=ax, palette=color)
    ax.axhline(y=true_ate, linestyle="--", color="black", label="True Effect")
    ax.set_ylabel("Estimated Treatment Effect on Log Income")
    st.pyplot(fig)