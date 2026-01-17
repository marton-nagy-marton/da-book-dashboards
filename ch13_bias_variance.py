import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

import warnings
warnings.filterwarnings('ignore')

color = ["#3a5e8c", "#10a53d", "#541352", "#ffcf20", "#2f9aa0"]

st.set_page_config(page_title='Bias-Variance Tradeoff', layout="wide")

# Streamlit app configuration
st.title("Digging Deep into the Bias-Variance Tradeoff with Polynomial Regression Simulations")
st.markdown("""
This interactive dashboard demonstrates the **bias-variance tradeoff** using **simulations with polynomial regression models**.
The logic of the simulation is as follows:

1. A **true data generating process** is defined as a **quartic polynomial function** with user-defined coefficients and added Gaussian noise.
2. A set number of **random samples** of a set size are **drawn from the data generating process**, and four polynomial **regression models of increasing complexity** (linear, quadratic, cubic, quartic) **are fitted to these simulated datasets**.
3. The dashboard visualizes:
    - The true model versus one random sample observed
    - The fitted models during the multiple simulations.
    - The distribution of estimated coefficients across simulations.
    - The bias-variance decomposition of the mean squared error (MSE) for each model.

**Each visualization helps you gain deeper insights** into how model complexity affects bias, variance, and overall prediction error.

### Things to observe:
- Looking at the first plot, notice how the observed data deviates from the true model due to noise. **What type of model would you choose to fit this data intuitively if you did not know the true underlying process?**
- In the second plot, observe how the different polynomial models fit the data across multiple simulations. **Find models that are consistently off the true model (high bias) versus those that vary widely across simulations (high variance)!**
- The third plot shows the distribution of a selected coefficient across simulations. **Validate your intuition about bias and variance by checking how far the estimated coefficients are from the true value and how spread out they are!**
- The bias-variance decomposition plot summarizes how bias and variance contribute to the overall MSE for each model. **Notice how simpler models tend to have higher bias but lower variance, while more complex models exhibit lower bias but higher variance** - so, there is indeed a tradeoff.
- Use the sidebar controls to adjust the data generating process, noise level, and simulation parameters to see how they impact the bias-variance tradeoff!
""")

# Sidebar inputs
st.sidebar.header("Settings")
st.sidebar.subheader("Data Generating Process")
with st.sidebar.expander("Adjust Coefficients"):
    true_intercept = st.slider("Intercept", min_value=-10.0, max_value=10.0, value=1.0, step=0.1)
    true_beta1 = st.slider("Linear Coefficient", min_value=-10.0, max_value=10.0, value=-0.5, step=0.1)
    true_beta2 = st.slider("Quadratic Coefficient", min_value=-10.0, max_value=10.0, value=4.5, step=0.1)
    true_beta3 = st.slider("Cubic Coefficient", min_value=-5.0, max_value=5.0, value=0.3, step=0.1)
    true_beta4 = st .slider("Quartic Coefficient", min_value=-2.0, max_value=2.0, value=-0.1, step=0.1)
    true_coefs = [true_intercept, true_beta1, true_beta2, true_beta3, true_beta4]
st.sidebar.subheader("Simulation Parameters")
noise_std = st.sidebar.slider("Noise Standard Deviation", min_value=0.0, max_value=1000.0, value=75.0, step=1.0)
num_iterations = st.sidebar.number_input("Number of Simulations", min_value=50, max_value=500, value=100, step=1)
sample_size = st.sidebar.number_input("Sample Size per Simulation", min_value=25, max_value=500, value=75, step=1)
x_min = st.sidebar.number_input("Minimum of X values", value=-5, step=1)
x_max = st.sidebar.number_input("Maximum of X values", value=5, step=1)
st.sidebar.subheader("Other")
random_seed = st.sidebar.number_input("Random Seed", min_value=0, value=42, step=1)

prng = np.random.default_rng(random_seed)

# Input x, output y based on betas
def trueModel(x):
    y = true_intercept + true_beta1 * x + true_beta2 * x**2 + true_beta3 * x**3 + true_beta4 * x**4
    return y

# Generate random data
def generateData(prng, sample_size, noise_std):
    x = np.linspace(x_min, x_max, sample_size)
    y_true = trueModel(x)
    y = y_true + prng.normal(0, noise_std, size=sample_size)

    feature_df = pd.DataFrame({'x': x})
    return feature_df, y

col1, col2 = st.columns(2)

# Visualize the true process and the observed data with noise
with col1:
    st.subheader("True Model vs. Observed Data with Noise")
    features, y = generateData(prng, sample_size, noise_std)
    fig, ax = plt.subplots()
    ax.plot(features['x'], trueModel(features['x']), label="Y without noise", color=color[1])
    ax.scatter(features['x'], y, label="Observed Y", c=color[0], s=20, alpha=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    sns.despine()
    fig.tight_layout()
    st.pyplot(fig)

# 4 models with increasing polynomial degrees, starting from linear
model_d1 = Pipeline([
    ('linreg', LinearRegression())
])
model_d2 = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('linreg', LinearRegression())
])
model_d3 = Pipeline([
    ('poly', PolynomialFeatures(degree=3, include_bias=False)),
    ('linreg', LinearRegression())
])
model_d4 = Pipeline([
    ('poly', PolynomialFeatures(degree=4, include_bias=False)),
    ('linreg', LinearRegression())
])

# Plot to visualize all fitted models during the Monte Carlo simulation
fig, axs = plt.subplots(2, 2, sharey=True, sharex=True)

betas = np.empty((num_iterations, 4, 5))  # (iterations, models, [intercept + 4 coefs])
all_predictions = np.empty((num_iterations, sample_size, 4))  # (iterations, samples, models)

# Perform the Monte Carlo simulation
for i in range(num_iterations):
    # Generate data
    features, y = generateData(prng, sample_size=sample_size, noise_std=noise_std)
    x_sorted = features.sort_values(by='x')
    x_sorted_vals = x_sorted['x']

    # Fit models
    model_d1 = model_d1.fit(features, y)
    model_d2 = model_d2.fit(features, y)
    model_d3 = model_d3.fit(features, y)
    model_d4 = model_d4.fit(features, y)

    # Predict on sorted x (sorting needed for plotting)
    d1_prediction = model_d1.predict(x_sorted)
    d2_prediction = model_d2.predict(x_sorted)
    d3_prediction = model_d3.predict(x_sorted)
    d4_prediction = model_d4.predict(x_sorted)

    # Store predictions
    all_predictions[i, :, 0] = d1_prediction
    all_predictions[i, :, 1] = d2_prediction
    all_predictions[i, :, 2] = d3_prediction
    all_predictions[i, :, 3] = d4_prediction

    models = [model_d1, model_d2, model_d3, model_d4]
    # Store coefficients
    for mnum, model in enumerate(models):
        betas[i, mnum, 0] = model.named_steps['linreg'].intercept_
        coefs = model.named_steps['linreg'].coef_
        for j in range(4):
            betas[i, mnum, j + 1] = coefs[j] if j < len(coefs) else np.nan

    # Plot current iteration predictions
    axs[0, 0].plot(x_sorted_vals, d1_prediction, color=color[0], alpha=0.025)
    sns.despine()
    axs[0, 1].plot(x_sorted_vals, d2_prediction, color=color[0], alpha=0.025)
    sns.despine()
    axs[1, 0].plot(x_sorted_vals, d3_prediction, color=color[0], alpha=0.025)
    sns.despine()
    axs[1, 1].plot(x_sorted_vals, d4_prediction, color=color[0], alpha=0.025)
    sns.despine()

# Compute ground truth values
true_y_sorted = trueModel(x_sorted_vals)  # shape: (sample_size,)

# Compute bias, variance, and MSE
mean_preds = np.mean(all_predictions, axis=0)
bias = mean_preds - true_y_sorted.to_numpy()[:, np.newaxis]
bias_squared = bias ** 2
variance = np.var(all_predictions, axis=0)
irreducible_error = noise_std ** 2
mse = np.mean((all_predictions - true_y_sorted.to_numpy()[:, np.newaxis]) ** 2, axis=0)

# Plot true model overlay for all models
for ax in axs.flat:
    ax.plot(x_sorted_vals, true_y_sorted, color=color[1], linestyle='dashed')

axs[0, 0].set_title("Linear Model")
axs[0, 1].set_title("Quadratic Model")
axs[1, 0].set_title("Cubic Model")
axs[1, 1].set_title("Quartic Model")
axs[0, 0].set_ylabel("Y")
axs[1, 0].set_ylabel("Y")
axs[1, 0].set_xlabel("X")
axs[1, 1].set_xlabel("X")
fig.tight_layout()

with col2:
    st.subheader("Predicted Models During Simulations")
    st.pyplot(fig)
    st.caption('Note: The dashed lines indicate the true model used in the data generating process (without noise).')

# Plot distribution of coefficients for the selected beta
model_labels = ["Linear Model", "Quadratic Model", "Cubic Model", "Quartic Model"]
coeff_names = ["Intercept", "$ß_1$ (coeff. on $x$)", "$β_2$ (coeff. on $x^2$)", "$β_3$ (coeff. on $x^3$)", "$β_4$(coeff. on $x^4$)"]

col1, col2 = st.columns(2)
with col1:
    st.subheader("Distribution of Selected Coefficient Across Simulations")
    beta_to_plot = st.segmented_control("Coefficient to Plot", 
        options=['Intercept', 'Linear', 'Quadratic', 'Cubic', 'Quartic'], 
        default='Linear'
    )
beta_to_plot = ['Intercept', 'Linear', 'Quadratic', 'Cubic', 'Quartic'].index(beta_to_plot)

# Determine which models contain the requested beta
included_models = [m for m in range(4) if beta_to_plot <= m + 1]  # model_d1 has 2 term, model_d2 has 3, etc.
n_models = len(included_models)

# Determine subplot layout
if n_models <= 2:
    nrows, ncols = 1, n_models
else:
    nrows, ncols = 2, math.ceil(n_models / 2)

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False, sharey=True, sharex=True)

# Plot the distribution of coefficients for each included model
for idx, model_idx in enumerate(included_models):
    row, col = divmod(idx, ncols)
    ax = axes[row][col]
    #sns.kdeplot(betas[:, model_idx, beta_to_plot], ax=ax, fill=True, color=color[0])
    sns.histplot(betas[:, model_idx, beta_to_plot], ax=ax, kde=False, color=color[0])
    ax.set_title(f"{model_labels[model_idx]}")
    ax.axvline(true_coefs[beta_to_plot], color='darkred', linestyle='dashed', label="True Coefficient", linewidth=3)
    ax.set_xlabel("Coefficient Value")
    ax.set_ylabel("Density")
    ax.legend().set_visible(False)
    sns.despine()

# Hide unused subplots if any
for idx in range(n_models, nrows * ncols):
    row, col = divmod(idx, ncols)
    axes[row][col].set_visible(False)

fig.tight_layout()

with col1:
    st.pyplot(fig)
    st.caption('Note: The dashed line indicates the true coefficient value used in the data generating process.')

model_names_var = ['model1', 'model2', 'model3', 'model4']

# Create dictionary for DataFrame
data = {}

# Fill dictionary with appropriately labeled columns
for i, name in enumerate(model_names_var):
    data[f'{name}_bias_squared'] = bias_squared[:, i]
    data[f'{name}_variance'] = variance[:, i]
    data[f'{name}_mse'] = mse[:, i]

# Create DataFrame
metrics_df = pd.DataFrame(data)

# Sum bias_squared and variance for each model across all samples
model_names = ['Linear Model', 'Quadratic Model', 'Cubic Model', 'Quartic Model']
bias_means = [metrics_df[f'{model}_bias_squared'].mean() for model in model_names_var]
var_means = [metrics_df[f'{model}_variance'].mean() for model in model_names_var]

# Plot stacked bar chart
x = np.arange(len(model_names))

with col2:
    st.subheader("Bias² and Variance Decomposition per Model")
    show_irred = st.checkbox("Show irreducible error on plot", value=False, key='irreducible_error_checkbox')
    st.markdown('#####')

fig, ax = plt.subplots()
ax.bar(x, bias_means, label='Bias²', color=color[0])
ax.bar(x, var_means, bottom=bias_means, label='Variance', color=color[1])
if show_irred:
    ax.bar(x, [irreducible_error]*len(model_names), bottom=np.array(bias_means)+np.array(var_means), label='Irreducible Error', color=color[3])

# Labels and titles
ax.set_xticks(x)
ax.set_xticklabels(model_names)
if show_irred:
    ax.set_ylabel('Mean Squared Error (MSE)')
else:
    ax.set_ylabel('Mean Squared Error (MSE) Less Irreducible Error')
ax.legend()
sns.despine()

plt.tight_layout()

with col2:
    st.pyplot(fig)
    st.caption('''
    Note: Bias and variance are computed as the mean of squared bias and variance across all samples for each model.
    ''')