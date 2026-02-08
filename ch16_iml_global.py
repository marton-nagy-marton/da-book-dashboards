import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from PyALE import ale

# Exact color palette
color = ["#3a5e8c", "#10a53d", "#541352", "#ffcf20", "#2f9aa0"]

# --- APP CONFIG ---
st.set_page_config(page_title="Interpretable ML Part 2", layout="wide")
st.title("Under the Hood of Global Interpretable Machine Learning: PDP, SHAP, and ALE")

# --- INTRODUCTION & USER GUIDE ---
st.markdown(r"""
This dashboard is designed to help you build an intuition for how different "Global" Interpretable ML methods interpret a model's behavior. 

Unlike local explanations (which focus on a single prediction), **Global Explanations** summarize how features affect the model's output across the entire dataset.
            
We are analyzing a **Black Box Model** (defined by you in the sidebar) that predicts an outcome based on three features.
For simplicity, we will use a linear model with an interaction term, but the methods we are demonstrating can be applied to any model, no matter how complex.
Note that on this dashboard, we are only interested in explaining how a model works - we are not dealing with how the model was trained. We take it as a given that the model is already trained and we are trying to understand its behavior.

#### How to use this dashboard:
1.  **Configure the Model:** Use the sidebar to change the coefficients ($\beta$). This changes how much weight the model gives to each feature and the interaction term.
2.  **Toggle Correlation:** Check "Correlate X1 and X2" to see how the methods behave when features are no longer independent. 
3.  **Analyze the Curves:** Switch between features ($X_1, X_2, X_3$) to see how different methods "attribute" importance.

#### Questions to ask yourself:
* **The Correlation Test:** When $X_1$ and $X_2$ are highly correlated, does the Partial Dependence Plot (PDP) look different from the SHAP dependency plot and the ALE curve? Why?
* **The Interaction Effect:** If you set the Interaction Term to a high value, how does the SHAP Dependency plot change its color mapping? Does the PDP capture this interaction effect, or does it miss it? What if you turn on the ICE curves?
""")
st.divider()

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.header("Model Configuration")
    beta1 = st.number_input(r"Coefficient for Feature 1 ($\beta_1$)", value=2.0)
    beta2 = st.number_input(r"Coefficient for Feature 2 ($\beta_2$)", value=3.0)
    beta3 = st.number_input(r"Coefficient for Feature 3 ($\beta_3$)", value=5.0)
    interaction = st.number_input(r"Interaction Term ($X_1*X_2$) Coefficient", value=1.0)
    
    st.header("Data Environment")
    correlated = st.checkbox("Correlate X1 and X2", value=False)
    
    st.header("Visualization Settings")
    target_feat_name = st.selectbox("Feature to Explain", ["X1", "X2", "X3"])
    show_ice = st.checkbox("Show ICE Curves (PDP)", value=True)
    shap_interaction = st.checkbox("SHAP: Best Interaction Coloring", value=True)

# --- DATA & MODEL ---
def get_data(is_correlated):
    np.random.seed(42)
    n = 500
    x1 = np.random.rand(n) * 10
    x3 = np.random.rand(n) * 10
    x2 = x1 + np.random.normal(0, 1, n) if is_correlated else np.random.rand(n) * 10
    return pd.DataFrame({'X1': x1, 'X2': x2, 'X3': x3})

X = get_data(correlated)

class UserModel:
    def predict(self, data):
        d = data.values if isinstance(data, pd.DataFrame) else data
        return beta1 * d[:,0] + beta2 * d[:,1] + beta3 * d[:,2] + interaction * (d[:,0] * d[:,1])

model = UserModel()

def finalize_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('white')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title("")
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)

# --- VISUALIZATION ---
col1, col2, col3 = st.columns(3)
plot_size = (5, 5)

# 1. PDP & ICE
with col1:
    st.markdown("##### Partial Dependence Plot")
    fig, ax = plt.subplots(figsize=plot_size)
    grid = np.linspace(X[target_feat_name].min(), X[target_feat_name].max(), 50)
    pdp_values = []
    
    for i in range(len(X)):
        instance = X.iloc[i:i+1].copy()
        temp_preds = [model.predict(instance.assign(**{target_feat_name: v}))[0] for v in grid]
        pdp_values.append(temp_preds)
        if show_ice:
            ax.plot(grid, temp_preds, color=color[1], alpha=0.1, linewidth=0.5)
    
    ax.plot(grid, np.mean(pdp_values, axis=0), color=color[0], linewidth=3)
    ax.set_xlabel(target_feat_name)
    ax.set_ylabel("Prediction")
    finalize_plot(ax)
    st.pyplot(fig)

# 2. SHAP Dependency
with col2:
    st.markdown("##### SHAP Dependency Plot")
    fig, ax = plt.subplots(figsize=plot_size)
    explainer = shap.Explainer(model.predict, X)
    shap_v = explainer(X)
    
    shap.dependence_plot(
        target_feat_name, shap_v.values, X,
        interaction_index='auto' if shap_interaction else None,
        ax=ax, show=False,
        cmap=plt.get_cmap("viridis") if shap_interaction else None
    )
    
    if not shap_interaction:
        for child in ax.get_children():
            if hasattr(child, 'set_color'):
                child.set_color(color[0])

    ax.set_xlabel(target_feat_name)
    ax.set_ylabel("SHAP Value")
    
    finalize_plot(ax)
    st.pyplot(fig)

# 3. ALE
with col3:
    st.markdown("##### ALE Curve")
    fig, ax = plt.subplots(figsize=plot_size)
    ale_eff = ale(X=X, model=model, feature=[target_feat_name], grid_size=20, 
                  plot=True, fig=fig, ax=ax, include_CI=False, feature_type='continuous')
    
    # Decisively remove rug plot elements
    for artist in ax.get_lines() + ax.collections:
        # Rug plots are usually lines with no width or specific markers
        if hasattr(artist, 'get_marker') and artist.get_marker() == '|':
            artist.remove()
        elif isinstance(artist, plt.Line2D) and artist.get_linewidth() < 1:
            # The main ALE line we set to 3 later; anything thin is likely the rug
            artist.remove()

    line = ax.get_lines()[0]
    line.set_color(color[0])
    line.set_linewidth(3)
    ax.set_xlabel(target_feat_name)
    ax.set_ylabel("Accumulated Local Effect")
    finalize_plot(ax)
    st.pyplot(fig)

col1, col2, col3 = st.columns(3)

with col1:
    with st.expander("🔍 Under the Hood: PDP & ICE"):
        st.markdown(r"""
        **Partial Dependence Plots (PDP)** show the marginal effect one or two features have on the predicted outcome of a machine learning model.
        
        **How it works:**
        It marginalizes the model output over the distribution of the other features ($C$). For a feature $x_s$:
        $$\hat{f}_{x_s}(x_s) = \frac{1}{n} \sum_{i=1}^{n} \hat{f}(x_s, x_{C}^{(i)})$$
        Essentially, it takes every row in your data, replaces the feature value with a fixed value (e.g., $X_1 = 5$), and averages the results.
        
        * **Pros:** Simple to implement and very intuitive. **ICE curves** (the faint lines) show if individual instances behave differently than the average.
        * **Cons:** If features are correlated, PDP creates "ghost" data points that are physically impossible (e.g., a 10-bedroom house that is only 200 sq. ft.), leading to biased explanations.
        """)

with col2:
    with st.expander("🔍 Under the Hood: SHAP"):
        st.markdown(r"""
        **SHAP (SHapley Additive exPlanations)** is based on cooperative game theory. It assigns each feature a "payout" (contribution) for a specific prediction.
        
        **How it works:**
        SHAP calculates the change in the expected model prediction when a feature is added to a set of features, averaged over all possible permutations of features.
        $$\phi_i = \sum_{S \subseteq \{x_1, \dots, x_p\} \setminus \{i\}} \frac{|S|!(p - |S| - 1)!}{p!} [f(S \cup \{i\}) - f(S)]$$
        
        * **Pros:** Solid theoretical foundation. The **Dependency Plot** is better than PDP because it doesn't just average - it can show the variance and interactions. Usually more accurate than PDP, especially when features are correlated.
        * **Cons:** Computationally expensive.
        """)

with col3:
    with st.expander("🔍 Under the Hood: ALE"):
        st.markdown(r"""
        **ALE** is the modern successor to PDP. it is specifically designed to handle correlated features.
        
        **How it works:**
        Instead of averaging over the entire dataset (which creates "ghost" points), ALE looks at small windows (intervals) of the feature. It calculates the **differences** in predictions for instances within that window and then accumulates (integrates) those differences.
        $$\hat{f}_{x_s, ALE}(x_s) = \int_{z_{min,1}}^{x_s} E_{X_C|X_s=z} \left[ \hat{f}^s(X_s, X_C) | X_s = z \right] dz$$
        
        * **Pros:** **Unbiased by correlation.** If $X_1$ and $X_2$ are correlated, ALE won't let $X_1$ "steal" the effects of $X_2$ through impossible data points.
        * **Cons:** Harder to explain to non-technical stakeholders. It requires a sufficient number of observations in each "bin" to be stable.
        """)