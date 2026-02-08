import streamlit as st
import numpy as np
import pandas as pd
from itertools import combinations
import math
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

color = ["#3a5e8c", "#10a53d", "#541352", "#ffcf20", "#2f9aa0"]

def calculate_shap_with_plot(model_predict, instance, background_data, feature_names=None):
    num_features = len(instance)
    feature_indices = list(range(num_features))
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(num_features)]
        
    reference_values = np.mean(background_data, axis=0)
    shap_values = np.zeros(num_features)

    def get_masked_prediction(subset_indices):
        input_data = reference_values.copy()
        for idx in subset_indices:
            input_data[idx] = instance[idx]
        return model_predict(input_data.reshape(1, -1))[0]

    # Process each feature one by one
    fig_results = []
    for i in feature_indices:
        subsets_data = [] # To store info for plotting
        remaining = [f for f in feature_indices if f != i]
        phi_i = 0
        
        for s_size in range(num_features):
            for S in combinations(remaining, s_size):
                weight = (math.factorial(len(S)) * math.factorial(num_features - len(S) - 1)) / math.factorial(num_features)
                
                val_with_i = get_masked_prediction(list(S) + [i])
                val_without_i = get_masked_prediction(S)
                marginal = val_with_i - val_without_i
                
                phi_i += weight * marginal
                
                # Store for plotting: (Subset label, Val With, Val Without, Weight, Marginal)
                subset_label = f"Subset: {', '.join(['X' + str(s+1) for s in S])}" if S else "Empty Set"
                subsets_data.append((subset_label, val_with_i, val_without_i, weight, marginal))
        
        shap_values[i] = phi_i
        
        # --- Plotting Logic for Feature i ---
        fig, ax = _plot_shap_contributions(feature_names[i], subsets_data, phi_i)
        fig_results.append((fig, ax))

    return shap_values, fig_results

def _plot_shap_contributions(feat_name, data, final_val):
    labels, with_i, without_i, weights, marginals = zip(*data)
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, with_i, width, label='With Feature', color=color[0], alpha=0.8)
    rects2 = ax.bar(x + width/2, without_i, width, label='Without Feature', color=color[1], alpha=0.8)
    
    ax.set_ylabel('Model Prediction')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, ha='center')
    ax.legend()

    # Add text elements for weight and marginal contribution
    for i, (w, m) in enumerate(zip(weights, marginals)):
        # Calculate height for text placement (top of the tallest bar in the group)
        height = max(with_i[i], without_i[i])
        ax.text(x[i], height, f'Weight: {w:.2f}\nContribution (Δ): {m:.2f}', 
                ha='center', va='bottom', fontsize=10, color='gray', fontweight='bold')
        
    #despine
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # move legend above plot in one row
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

    plt.tight_layout()
    return fig, ax

def explain_lime_with_plot(model_predict, instance, background_data, feature_names=None):
    num_features = len(instance)
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(num_features)]
    
    # 1. Setup Scaling and Perturbations
    feature_sd = np.std(background_data, axis=0)
    feature_sd = np.where(feature_sd == 0, 1e-6, feature_sd)
    
    num_samples = 1000
    noise = np.random.normal(0, 1, (num_samples, num_features))
    perturbed_samples = instance + (noise * feature_sd)
    perturbed_samples[0, :] = instance # Ensure original is present
    
    # 2. Get Model Predictions and Weights
    y_perturbed = model_predict(perturbed_samples)
    
    scaled_instance = instance / feature_sd
    scaled_perturbed = perturbed_samples / feature_sd
    distances = pairwise_distances(scaled_perturbed, scaled_instance.reshape(1, -1)).flatten()
    
    kernel_width = 0.75 * np.sqrt(num_features)
    weights = np.exp(-(distances**2) / (kernel_width**2))
    
    # 3. Fit Local Model (The explanation)
    local_model = Ridge(alpha=1.0)
    local_model.fit(perturbed_samples, y_perturbed, sample_weight=weights)
    betas = local_model.coef_
    intercept = local_model.intercept_

    # 4. Plotting for each feature
    y_background = model_predict(background_data)
    
    plot_results = []
    for i in range(num_features):
        plt.figure(figsize=(12, 5.6))
        
        # Plot Global Background Data
        plt.scatter(background_data[:, i], y_background, color='orange', label='Background Data', s=15, marker='^', alpha=0.6)
        
        # Plot Perturbed Samples (Color intensity by weight)
        sc = plt.scatter(perturbed_samples[:100, i], y_perturbed[:100], c=weights[:100], cmap='viridis', 
                         alpha=0.6, s=20, label='Perturbed Samples')
        plt.colorbar(sc, label='LIME Weight')
        
        # Highlight the Instance to explain
        plt.scatter(instance[i], model_predict(instance.reshape(1, -1)), 
                    color=color[0], edgecolors='black', s=250, marker='*', label='Instance to Explain', zorder=5)
        
        # Draw the local regression line
        # Note: To plot a 2D line for a multidimensional model, we vary feature i 
        # while keeping other features at the instance's values.
        x_range = np.linspace(perturbed_samples[:100, i].min(), perturbed_samples[:100, i].max(), 100)
        # Prediction: y = beta_i * x_i + [sum(beta_j * instance_j) + intercept]
        other_features_contribution = np.sum(betas * instance) - (betas[i] * instance[i]) + intercept
        y_line = (betas[i] * x_range) + other_features_contribution
        
        plt.plot(x_range, y_line, color=color[0], linestyle='--', linewidth=2, label='Local Linear Fit')
        
        # Labels and Metadata
        plt.ylabel("Model Prediction")
        plt.legend()

        #despine
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        #move legend above plot in two rows
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
        
        # Add Beta Text
        plt.text(0.05, 0.95, f"Local Coefficient: {betas[i]:.2f}", 
                 transform=plt.gca().transAxes, fontsize=10, fontweight='bold',
                 color='gray',
                 verticalalignment='top')
        
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plot_results.append(plt.gcf())

    return betas, plot_results

st.set_page_config(page_title="Interpretable ML Part 1", layout="wide")
st.title("Under the Hood of Local Interpretable Machine Learning: SHAP & LIME")
st.markdown(r"""
### Introduction & User Guide

This tool is designed to demystify how complex machine learning models make decisions by visualizing two of the most popular explanation methods: **SHAP** and **LIME**.

#### What is this explaining?
We are analyzing a **Black Box Model** (defined by you in the sidebar) that predicts an outcome based on three features.
For simplicity, we will use a linear model with an interaction term, but the methods we are demonstrating can be applied to any model, no matter how complex.
Note that on this dashboard, we are only interested in explaining how a model works - we are not dealing with how the model was trained. We take it as a given that the model is already trained and we are trying to understand its behavior.
            
Also note that these methods require a background dataset to compare against. For simplicity, we generate a background set of 100 samples with all feature values distributed between 0 and 10.

* **SHAP (SHapley Additive exPlanations):** Acts like a "fairness" accountant. It treats the final prediction as a "payout" and uses game theory to distribute that value among the features based on their average marginal contribution across all possible feature combinations.
* **LIME (Local Interpretable Model-agnostic Explanations):** Acts like a "local" surveyor. It zooms in on your specific data point, generates a "cloud" of perturbed samples nearby, and fits a simple linear model to that neighborhood to see how the prediction changes.

#### How to use this dashboard
1.  **Define the "Truth" (Sidebar):** Adjust the coefficients ($\beta$) and the Interaction Term. This creates the "hidden" logic of the model.
2.  **Pick an Instance (Sidebar):** Choose the specific $X$ values you want to explain.
3.  **Analyze SHAP (Left Column):**
    * Look at the **Base Value** (the average prediction of the model).
    * Examine the **Marginal Contribution** plots to see how Feature $i$ behaves when added to different "coalitions" of other features.
4.  **Analyze LIME (Right Column):** 
    * Observe the **Local Linear Fit**. The dashed line shows how LIME approximates the model's slope at your chosen point.
    * Compare the **Local Beta** (LIME's guess) to the **Analytical** (Mathematical Truth) to see how accurate the approximation is.

#### Guiding Questions for Discovery
While you interact with the charts, ask yourself:

* **On Interactions:** How does changing the $X_1 \cdot X_2$ Interaction coefficient change the SHAP values for Feature 1? *Hint: The interaction effect should be split between the two participating features.*
* **On Locality:** If you change an instance value to an extreme (e.g., $X_1=10$), does the LIME linear fit still look accurate, or does the model's non-linearity make the local explanation less reliable? What about such perturbed data points that are far from the background data distribution?
* **On Reality vs. Approximation:** How close are the LIME local coefficients to the true analytical coefficients for this linear model?
* **On Agreement:** Do SHAP and LIME agree on which feature is "most important" for your current settings? If they differ, why might that be?
            
---
    """)

with st.sidebar:
    st.header("Model and Instance Configuration")
    
    # Define a simple linear model with interaction
    st.subheader("Model Definition")
    beta1 = st.number_input(r"Coefficient for Feature 1 ($\beta_1$)", value=2.0)
    beta2 = st.number_input(r"Coefficient for Feature 2 ($\beta_2$)", value=3.0)
    beta3 = st.number_input(r"Coefficient for Feature 3 ($\beta_3$)", value=5.0)
    interaction = st.number_input(r"Interaction Term ($X_1*X_2$) Coefficient", value=1.0)
    
    # Input instance values
    st.subheader("Instance to Explain")
    feature_1 = st.number_input("Feature 1 ($X_1$)", value=4.0, max_value=10.0, min_value=0.0, step=0.1)
    feature_2 = st.number_input("Feature 2 ($X_2$)", value=9.0, max_value=10.0, min_value=0.0, step=0.1)
    feature_3 = st.number_input("Feature 3 ($X_3$)", value=7.0, max_value=10.0, min_value=0.0, step=0.1)
    
    instance = np.array([feature_1, feature_2, feature_3])

def true_model(X):
    return beta1 * X[0] + beta2 * X[1] + beta3 * X[2] + interaction * X[0] * X[1]

def model_predict(X):
    return np.array([true_model(x) for x in X])

def data_generation(num_samples=100):
    np.random.seed(42)
    X = np.random.rand(num_samples, 3) * 10
    return X

background_data = data_generation(num_samples=100)

# Create two primary columns for SHAP and LIME
col_shap, col_lime = st.columns(2, gap="large")

col_shap2, col_lime2 = st.columns(2, gap="large")

# --- SHAP SECTION ---
with col_shap:
    st.markdown("#### SHAP Explanation")
    
    # Logic
    shap_values, shap_figs = calculate_shap_with_plot(
        model_predict, instance, background_data, 
        feature_names=["Feature 1", "Feature 2", "Feature 3"]
    )
    base_val = np.mean(model_predict(background_data))
    prediction = np.sum(shap_values) + base_val

    # Summary Metrics
    m1, m2 = st.columns(2)
    m1.metric("Base Value", f"{base_val:.2f}")
    m2.metric("Final Prediction", f"{prediction:.2f}", delta=f"{np.sum(shap_values):.2f}")

    # Feature Breakdown in a clean table or list
    with st.expander("View Feature SHAP Values", expanded=True):
        for i, val in enumerate(shap_values):
            st.write(f"**Feature {i+1}:** {val:.2f}")
    
    st.markdown("""
    The SHAP values represent the contribution of each feature to the difference between the model's prediction for the instance and the average prediction across the background data.
    In this case, the SHAP values indicate how much each feature is pushing the prediction away from the average prediction, with positive values indicating a push towards a higher prediction and negative values indicating a push towards a lower prediction.
    """)

with col_shap2:
    # Visualization Tab
    tab_plot, tab_math = st.tabs(["Plots", "Theory"])
    with tab_plot:
        for i, (fig, ax) in enumerate(shap_figs):
            st.caption(f"Feature {i+1} Contribution")
            st.pyplot(fig)
            
    with tab_math:
        st.markdown(r"""
        The SHAP value for a feature i is calculated using the following formula:
$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! (|N| - |S| - 1)!}{|N|!} \left[ f_{S \cup \{i\}}(x_{S \cup \{i\}}) - f_S(x_S) \right]$$
        """)
        
        st.markdown(r"""
        Where:
        - $N$ is the set of all features.
        - $S$ is a subset of features that does not include feature $i$.
        - $f_{S \cup \{i\}}(x_{S \cup \{i\}})$ is the model's prediction when feature $i$ is included in the subset $S$.
        - $f_S(x_S)$ is the model's prediction when feature $i$ is excluded from the subset $S$. The exclusion is typically handled by replacing the feature's value with a baseline (often the average value from the background data).
        
        This formula essentially averages the marginal contribution of feature $i$ across all possible subsets of features, weighted by the number of features in the subset.
        """)

# --- LIME SECTION ---
with col_lime:
    st.markdown("#### LIME Explanation")
    
    # Logic
    lime_betas, lime_figs = explain_lime_with_plot(
        model_predict, instance, background_data, 
        feature_names=["Feature 1", "Feature 2", "Feature 3"]
    )

    # Local Coefficients Table
    st.markdown("##### Local Linear Model Coefficients")
    
    # Using a table for direct comparison is often cleaner than a bullet list
    comparison_data = {
        "Feature": ["F1", "F2", "F3"],
        "Local Beta": [f"{b:.2f}" for b in lime_betas],
        "Analytical": [
            f"{beta1 + interaction * instance[1]:.2f}",
            f"{beta2 + interaction * instance[0]:.2f}",
            f"{beta3:.2f}"
        ]
    }
    st.table(comparison_data)

    st.markdown(f"""    
    These coefficients represent the local linear approximation of the model's behavior around the instance being explained.
    In fact, this is an approximation of the partial derivatives of the model's prediction with respect to each feature at the instance 
    (which can be easily calculated for a linear model, but not so much for more complex black box models, thus the need for methods like LIME). 
                """)

with col_lime2:
    # Visualization Tab
    tab_lime_plot, tab_details = st.tabs(["Plots", "Theory"])
    with tab_lime_plot:
        for i, fig in enumerate(lime_figs):
            st.caption(f"LIME Explanation Plot: Feature {i+1}")
            st.pyplot(fig)
            
    with tab_details:
        st.markdown("""
        LIME works by perturbing the input data around the instance of interest and observing how the model's predictions change in response to these perturbations. 
        
        It then fits a simple, interpretable model (like a linear regression) to these perturbed samples, weighted by their proximity to the original instance. 
        
        The coefficients of this local model serve as explanations for the importance of each feature in the prediction for that specific instance.
        """)






