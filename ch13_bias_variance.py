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

st.set_page_config(page_title='Ch. 13 - Bias-Variance Tradeoff', layout="wide")

# Streamlit app configuration
st.title("Ch. 13 - Bias-Variance Tradeoff")
st.markdown("""
This interactive dashboard demonstrates the **bias-variance tradeoff** using polynomial regression.
""")

# Sidebar inputs
st.sidebar.header("Settings")
st.sidebar.subheader("Data Generating Process")
true_intercept = st.sidebar.slider("True Intercept", min_value=-10.0, max_value=10.0, value=1.5, step=0.1)
true_beta1 = st.sidebar.slider("True Beta 1", min_value=-10.0, max_value=10.0, value=1.0, step=0.1)
true_beta2 = st.sidebar.slider("True Beta 2", min_value=-10.0, max_value=10.0, value=2.0, step=0.1)
true_beta3 = st.sidebar.slider("True Beta 3", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
true_beta4 = st.sidebar.slider("True Beta 4", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
true_coefs = [true_intercept, true_beta1, true_beta2, true_beta3, true_beta4]
noise_std = st.sidebar.slider("Noise Standard Deviation", min_value=0.0, max_value=1000.0, value=5.0, step=1.0)
st.sidebar.subheader("Simulation Parameters")
num_iterations = st.sidebar.number_input("Number of Simulations", min_value=50, max_value=1000, value=100, step=1)
sample_size = st.sidebar.number_input("Sample Size per Simulation", min_value=50, max_value=1000, value=250, step=1)
st.sidebar.subheader('Plotting parameters')
x_min = st.sidebar.number_input("X Min", value=-10, step=1)
x_max = st.sidebar.number_input("X Max", value=10, step=1)
beta_to_plot = st.sidebar.selectbox("Beta to Plot", 
    options=['Intercept', 'Beta1', 'Beta2', 'Beta3', 'Beta4'], 
    index=0
)
beta_to_plot = ['Intercept', 'Beta1', 'Beta2', 'Beta3', 'Beta4'].index(beta_to_plot)
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
    ax.plot(features['x'], trueModel(features['x']), label="True Y", color=color[1])
    ax.scatter(features['x'], y, label="Observed Y", c=color[0], s=20, alpha=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
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
    axs[0, 1].plot(x_sorted_vals, d2_prediction, color=color[0], alpha=0.025)
    axs[1, 0].plot(x_sorted_vals, d3_prediction, color=color[0], alpha=0.025)
    axs[1, 1].plot(x_sorted_vals, d4_prediction, color=color[0], alpha=0.025)

# Compute ground truth values
true_y_sorted = trueModel(x_sorted_vals)  # shape: (sample_size,)

# Compute bias, variance, and MSE
mean_preds = np.mean(all_predictions, axis=0)
bias = mean_preds - true_y_sorted.to_numpy()[:, np.newaxis]
bias_squared = bias ** 2
variance = np.var(all_predictions, axis=0)
mse = np.mean((all_predictions - true_y_sorted.to_numpy()[:, np.newaxis]) ** 2, axis=0)

# Plot true model overlay for all models
for ax in axs.flat:
    ax.plot(x_sorted_vals, true_y_sorted, color=color[1], linestyle='dashed')

axs[0, 0].set_title("Linear model")
axs[0, 1].set_title("Quadratic model")
axs[1, 0].set_title("Cubic model")
axs[1, 1].set_title("Quartic model")
axs[0, 0].set_ylabel("Y")
axs[1, 0].set_ylabel("Y")
axs[1, 0].set_xlabel("X")
axs[1, 1].set_xlabel("X")
fig.tight_layout()

with col2:
    st.subheader("Predicted Models During Simulations")
    st.pyplot(fig)

# Plot distribution of coefficients for the selected beta
model_labels = ["Linear Model", "Quadratic Model", "Cubic Model", "Quartic Model"]
coeff_names = ["Intercept", "$ß_1$ (coeff. on $x$)", "$β_2$ (coeff. on $x^2$)", "$β_3$ (coeff. on $x^3$)", "$β_4$(coeff. on $x^4$)"]

# Determine which models contain the requested beta
included_models = [m for m in range(4) if beta_to_plot <= m + 1]  # model_d1 has 2 term, model_d2 has 3, etc.
n_models = len(included_models)

# Determine subplot layout
if n_models <= 3:
    nrows, ncols = 1, n_models
else:
    nrows, ncols = 2, math.ceil(n_models / 2)

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False, sharey=True, sharex=True)

# Plot the distribution of coefficients for each included model
for idx, model_idx in enumerate(included_models):
    row, col = divmod(idx, ncols)
    ax = axes[row][col]
    sns.kdeplot(betas[:, model_idx, beta_to_plot], ax=ax, fill=True, color=color[0])
    ax.set_title(f"{model_labels[model_idx]}: {coeff_names[beta_to_plot]}")
    ax.axvline(true_coefs[beta_to_plot], color='darkred', linestyle='dashed', label="True Coefficient")
    ax.set_xlabel("Coefficient Value")
    ax.set_ylabel("Density")
    ax.legend()

# Hide unused subplots if any
for idx in range(n_models, nrows * ncols):
    row, col = divmod(idx, ncols)
    axes[row][col].set_visible(False)

fig.tight_layout()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Distribution of Selected Coefficient Across Simulations")
    st.pyplot(fig)

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
bias_sums = [metrics_df[f'{model}_bias_squared'].sum() for model in model_names_var]
var_sums = [metrics_df[f'{model}_variance'].sum() for model in model_names_var]

# Plot stacked bar chart
x = np.arange(len(model_names))

fig, ax = plt.subplots()
ax.bar(x, bias_sums, label='Bias²', color=color[0])
ax.bar(x, var_sums, bottom=bias_sums, label='Variance', color=color[1])

# Labels and titles
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_ylabel('Mean Squared Error (MSE)')
ax.legend()

plt.tight_layout()

with col2:
    st.subheader("Bias² and Variance Decomposition per Model")
    st.pyplot(fig)
    st.write('''
    Note: Bias and variance are computed as the sum of squared bias and variance across all samples for each model.
    ''')