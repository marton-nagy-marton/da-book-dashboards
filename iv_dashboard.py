import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf
import seaborn as sns
import streamlit as st
from scipy import stats

color = ["#3a5e8c", "#10a53d", "#541352", "#ffcf20", "#2f9aa0"]

st.set_page_config(page_title="Instrumental Variables", layout="wide")
sns.set_theme(style="whitegrid")

VARIABLE_LABELS = {
    "age": "Age",
    "female": "Female",
    "education_years": "Years of Education",
    "prior_experience_months": "Prior Work Experience (Months)",
}

GROUP_LABELS = {0: "Assigned Control Group", 1: "Assigned Treatment Group"}


def generate_synthetic_data(
    n: int,
    assignment_rate: float,
    target_p: float,
    target_q: float,
    true_late: float,
    base_earn: float,
    gamma_motiv_train: float,
    gamma_motiv_earn: float,
    seed: int,
) -> tuple[pd.DataFrame, float]:

    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(35, 8, n), 20, 60)
    female = rng.binomial(1, 0.45, n)
    education_years = np.clip(rng.normal(12, 2.5, n), 8, 20)
    prior_experience_months = rng.gamma(2.0, 18.0, n)
    motivation = rng.normal(0, 1, n)

    assigned = rng.binomial(1, assignment_rate, n)

    # ---- First stage (latent normal index) ----
    scale = np.sqrt(1 + gamma_motiv_train**2)
    eps = 1e-12
    p_safe = np.clip(target_p, eps, 1 - eps)
    q_safe = np.clip(target_q, eps, 1 - eps)

    alpha_p = stats.norm.ppf(p_safe) * scale
    alpha_q = stats.norm.ppf(q_safe) * scale

    epsilon = rng.normal(0, 1, n)

    latent = (
        (alpha_p * assigned + alpha_q * (1 - assigned))
        + gamma_motiv_train * motivation
        + epsilon
    )

    trained = (latent > 0).astype(int)

    # ---- Outcome ----
    earnings = (
        base_earn
        + true_late * trained
        + gamma_motiv_earn * motivation
        + 0.05 * (age - 35)
        + 0.20 * (education_years - 12)
        + 0.075 * (prior_experience_months - 36)
        + 0.1 * female
        + rng.normal(0, 0.15, n)
    )

    return pd.DataFrame({
        "person_id": np.arange(1, n + 1),
        "assigned": assigned,
        "trained": trained,
        "earnings": earnings,
        "age": age,
        "female": female,
        "education_years": education_years,
        "prior_experience_months": prior_experience_months,
        "motivation": motivation
    })

def compute_balance_table(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    rows = []
    for var in variables:
        treat = df.loc[df["assigned"] == 1, var]
        ctrl = df.loc[df["assigned"] == 0, var]
        t_stat, p_val = stats.ttest_ind(treat, ctrl, equal_var=False, nan_policy="omit")
        rows.append(
            {
                "Variable": VARIABLE_LABELS[var],
                "Treatment mean": treat.mean(),
                "Control mean": ctrl.mean(),
                "Difference": treat.mean() - ctrl.mean(),
                "t-stat": t_stat,
                "p-value": p_val,
            }
        )

    out = pd.DataFrame(rows)
    out.set_index("Variable", inplace=True)
    return out


def model_term_stats(result, term: str) -> dict:
    params = result.coef()
    bse = result.se()
    pvals = result.pvalue()

    coef = float(params[term])
    se = float(bse[term])
    pval = float(pvals[term])
    ci_low = coef - 1.96 * se
    ci_high = coef + 1.96 * se

    return {
        "coef": coef,
        "se": se,
        "pval": pval,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def run_models(df: pd.DataFrame, controls: list[str]) -> tuple[dict, pd.DataFrame]:
    ols_naive = pf.feols("earnings ~ trained", data=df, vcov="HC1")
    reduced_form = pf.feols("earnings ~ assigned", data=df, vcov="HC1")
    first_stage = pf.feols("trained ~ assigned", data=df, vcov="HC1")
    # add handling for IV when there is perfect compliance -> singular matrix error
    try:
        iv_basic = pf.feols("earnings ~ 1 | trained ~ assigned", data=df, vcov="HC1")
    except np.linalg.LinAlgError:
        iv_basic = ols_naive  # fallback to naive OLS if IV cannot be estimated (e.g. perfect compliance)
    iv_controls_formula = "earnings ~ " + " + ".join(controls) + " | trained ~ assigned"
    iv_ctrl = pf.feols(iv_controls_formula, data=df, vcov="HC1")

    aitte = float(reduced_form.coef()["assigned"])
    fs_coef = float(first_stage.coef()["assigned"])
    wald = aitte / fs_coef if fs_coef != 0 else np.nan

    naive_stats = model_term_stats(ols_naive, "trained")
    rf_stats = model_term_stats(reduced_form, "assigned")

    iv_basic_stats = model_term_stats(iv_basic, "trained")
    iv_ctrl_stats = model_term_stats(iv_ctrl, "trained")

    rows = [
        {
            "Model": "Naive OLS",
            "Term": "trained",
            "Estimate": naive_stats["coef"],
            "SE": naive_stats["se"],
            "p-value": naive_stats["pval"],
            "95% CI low": naive_stats["ci_low"],
            "95% CI high": naive_stats["ci_high"],
            "Interpretation": "Biased association: actual training vs employment",
        },
        {
            "Model": "AITTE (Reduced Form)",
            "Term": "assigned",
            "Estimate": rf_stats["coef"],
            "SE": rf_stats["se"],
            "p-value": rf_stats["pval"],
            "95% CI low": rf_stats["ci_low"],
            "95% CI high": rf_stats["ci_high"],
            "Interpretation": "Intent-to-treat effect of assignment",
        },
        {
            "Model": "IV / 2SLS",
            "Term": "trained",
            "Estimate": iv_basic_stats["coef"],
            "SE": iv_basic_stats["se"],
            "p-value": iv_basic_stats["pval"],
            "95% CI low": iv_basic_stats["ci_low"],
            "95% CI high": iv_basic_stats["ci_high"],
            "Interpretation": "LATE for compliers (using assignment as instrument)",
        },
        {
            "Model": "IV with controls",
            "Term": "trained",
            "Estimate": iv_ctrl_stats["coef"],
            "SE": iv_ctrl_stats["se"],
            "p-value": iv_ctrl_stats["pval"],
            "95% CI low": iv_ctrl_stats["ci_low"],
            "95% CI high": iv_ctrl_stats["ci_high"],
            "Interpretation": "LATE with covariate adjustment",
        },
    ]

    summary = {
        "aitte": aitte,
        "first_stage_coef": fs_coef,
        "wald": wald,
    }
    return summary, pd.DataFrame(rows)


def compliance_table(df: pd.DataFrame) -> pd.DataFrame:
    tab = pd.crosstab(df["trained"], df["assigned"])
    tab = tab.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    tab.index = ["Not Trained", "Trained"]
    tab.columns = ["Assigned Control Group", "Assigned Treatment Group"]
    tab["Total"] = tab.sum(axis=1)

    totals = pd.DataFrame([tab.sum(axis=0)], index=["Total"])
    out = pd.concat([tab, totals])

    grand_total = len(df)
    return out.applymap(lambda val: f"{int(val)} ({val / grand_total:.1%})")


def build_dgp_graphviz(params: dict) -> str:
    return f"""
digraph DGP {{
    rankdir=LR;
    graph [fontname="static/segoeui", fontsize=12];
    node [shape=box, style="rounded,filled", fillcolor="{color[0] + '25'}", color="{color[0]}", fontname="static/segoeui"];
    edge [fontname="static/segoeui", color="#475569"];

    assigned [label="Random Assignment\n(Instrument)", fillcolor="{color[1] + '25'}", color="{color[1]}"];
    trained [label="Actual Participation\n(Treatment)", fillcolor="{color[1] + '25'}", color="{color[1]}"];
    earnings [label="Earnings Outcome", fillcolor="{color[1] + '25'}", color="{color[1]}"];
    motivation [label="Motivation\n(Unobserved Confounder)", fillcolor="{color[2] + '25'}", color="{color[2]}"];
    covariates [label="Other Observed\nCovariates"];

    assigned -> trained [color="{color[1]}", penwidth=3];
    trained -> earnings [label="Direct treatment effect = {params['true_late']:.2f}", color="{color[1]}", penwidth=3];
    assigned -> earnings [label="Reduced form (assignment effect)", style=dashed, color="{color[1]}"];

    motivation -> trained [label="Motivation's effect on participation = {params['gamma_motiv_train']:.2f}", color="{color[2]}"];
    motivation -> earnings [label="Motivation confounding = {params['gamma_motiv_earn']:.2f}", color="{color[2]}"];
    covariates -> earnings [label="Effect of other covariates", color="{color[0]}"];
}}
"""


def build_iv_graphviz(params: dict) -> str:
    return f"""digraph IV {{
    rankdir=TB;
    graph [fontname="static/segoeui", fontsize=12];
    node [shape=box, style="rounded,filled", fillcolor="{color[0] + '25'}", color="{color[0]}", fontname="static/segoeui"];
    edge [fontname="static/segoeui", color="#475569"];
    assigned [label="Random Assignment\n(Instrument)", fillcolor="{color[0] + '25'}", color="{color[0]}"];
    trained [label="Actual Participation\n(Treatment)", fillcolor="{color[0] + '25'}", color="{color[0]}"];
    earnings [label="Earnings Outcome", fillcolor="{color[0] + '25'}", color="{color[0]}"];
    assigned -> trained [label="First stage estimate = {params['first_stage_coef']:.2f}", color="{color[0]}", penwidth=3];
    trained -> earnings [label="IV LATE estimate = {params['iv_late']:.2f}", color="{color[0]}", penwidth=3];
    assigned -> earnings [label="Reduced form\n(AITTE)\nestimate = {params['aitte']:.2f}", style=dashed, color="{color[0]}"];
}}"""


def plot_balance_grid(df: pd.DataFrame):
    plot_df = df.copy()
    plot_df["Assignment Group"] = plot_df["assigned"].map(GROUP_LABELS)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    continuous_specs = [
        ("age", "Age (years)"),
        ("education_years", "Years of education"),
        ("prior_experience_months", "Prior work experience (months)"),
    ]

    for idx, (var, x_label) in enumerate(continuous_specs):
        ax = axes.flat[idx]

        sns.kdeplot(
            data=plot_df,
            x=var,
            hue="Assignment Group",
            common_norm=False,
            fill=True,
            alpha=0.25,
            palette=[color[0], color[1]],
            linewidth=2,
            ax=ax,
        )
        ax.set_xlabel(x_label, fontsize=16)
        ax.set_ylabel("Density", fontsize=16)
        ax.tick_params(axis="both", labelsize=14)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        sns.despine(ax=ax)

    female_plot = plot_df.groupby(["assigned", "female"]).size().reset_index(name="Count")
    female_plot["Assignment Group"] = female_plot["assigned"].map(GROUP_LABELS)
    female_plot["Female Value"] = female_plot["female"].map({0: "0 = No", 1: "1 = Yes"})

    ax = axes[1, 1]
    sns.barplot(
        data=female_plot,
        x="Female Value",
        y="Count",
        hue="Assignment Group",
        palette=[color[0], color[1]],
        ax=ax,
    )
    ax.set_xlabel("Female indicator value", fontsize=16)
    ax.set_ylabel("Number of participants", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    sns.despine(ax=ax)

    handles = [
        plt.Line2D([0], [0], color=color[0], lw=4, label="Assigned Control Group"),
        plt.Line2D([0], [0], color=color[1], lw=4, label="Assigned Treatment Group"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=16,
        bbox_to_anchor=(0.5, 1.05),
    )

    plt.tight_layout()
    return fig


def plot_estimator_comparison(results_df: pd.DataFrame, true_late: float):
    fig, ax = plt.subplots(figsize=(10, 7))
    ordered = results_df.copy().reset_index(drop=True)
    ordered = ordered.iloc[[3, 2, 1, 0]]
    y_pos = np.arange(len(ordered))

    estimates = ordered["Estimate"].to_numpy()
    ci_low = ordered["95% CI low"].to_numpy()
    ci_high = ordered["95% CI high"].to_numpy()
    err_low = estimates - ci_low
    err_high = ci_high - estimates

    ax.errorbar(
        estimates,
        y_pos,
        xerr=[err_low, err_high],
        fmt="o",
        markersize=12,
        color=color[0],
        ecolor=color[1],
        elinewidth=4,
        capsize=6,
    )
    ax.axvline(true_late, linestyle=":", color=color[2], linewidth=4, label="True effect")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered["Model"], fontsize=16)
    ax.set_xlabel("Estimated effect on earnings (thousand USD)", fontsize=16)
    ax.tick_params(axis="x", labelsize=14)
    sns.despine(ax=ax)
    plt.tight_layout()
    return fig


def map_simple_to_advanced(
    n: int,
    non_compliance: str,
    self_selection: str,
    motivation_participation: str,
    motivation_confounding: str,
) -> dict:
    non_compliance_to_p = {"None": 1.00, "Low": 0.85, "Moderate": 0.60, "High": 0.50}
    self_selection_to_q = {"None": 0.00, "Low": 0.15, "Moderate": 0.35, "High": 0.50}
    motivation_to_train = {"None": 0.00, "Low": 0.30, "Moderate": 0.60, "High": 0.80}
    motivation_to_employ = {"None": 0.00, "Low": 1.00, "Moderate": 2.00, "High": 4.00}

    target_q = self_selection_to_q[self_selection]
    target_p = non_compliance_to_p[non_compliance]

    return {
        "n": n,
        "assignment_rate": 0.50,
        "target_p": target_p,
        "target_q": target_q,
        "true_late": 1.00,
        "base_earn": 6.00,
        "gamma_motiv_train": motivation_to_train[motivation_participation],
        "gamma_motiv_earn": motivation_to_employ[motivation_confounding],
        "seed": 42,
    }


def render_shared_sidebar() -> tuple[str, dict, st.delta_generator.DeltaGenerator]:
    with st.sidebar:
        st.header('Navigation')
        page = st.radio(
            "",
            ["Introduction", "Data balance and diagnostics", "Regression analysis"],
            index=0,
            label_visibility="collapsed",
        )
        st.header("Simulation Controls")
        advanced_settings = st.segmented_control(
            "Type of settings",
            options=["Simple", "Advanced"],
            default="Simple",
            help="Simple mode maps intuitive descriptions to the underlying parameters; advanced mode allows direct control over all parameters.",
        )

        if advanced_settings == "Simple":
            n = st.slider(
                "Number of participants",
                min_value=200,
                max_value=20_000,
                value=5_000,
                step=100,
                help="Total sample size in the simulated study.",
            )
            non_compliance = st.segmented_control(
                "How strong is non-compliance in the treatment group?",
                options=["None", "Low", "Moderate", "High"],
                default="Low",
                help="Share of assigned participants who do not participate in the training (non-compliance rate).",
            )
            self_selection = st.segmented_control(
                "How strong is self-selection in the control group?",
                options=["None", "Low", "Moderate", "High"],
                default="Low",
                help="Share of control participants who still participate in the training (self-selection rate).",
            )

            motivation_participation = st.segmented_control(
                "How strong is the effect of motivation on participation?",
                options=["None", "Low", "Moderate", "High"],
                default="Moderate",
                help="How much motivation increases the probability of participating in the training. This creates confounding bias for naive comparisons.",
            )
            motivation_confounding = st.segmented_control(
                "How strong is the effect of motivation on earnings?",
                options=["None", "Low", "Moderate", "High"],
                default="Moderate",
                help="How much motivation increases earnings independently of training. This creates confounding bias for naive comparisons.",
            )

            params = map_simple_to_advanced(
                n=n,
                non_compliance=non_compliance,
                self_selection=self_selection,
                motivation_participation=motivation_participation,
                motivation_confounding=motivation_confounding,
            )
        else:
            n = st.slider(
                "Number of participants",
                min_value=200,
                max_value=20_000,
                value=5_000,
                step=100,
                help="Total sample size in the simulated study.",
            )

            st.subheader("Assignment and participation")
            assignment_rate = st.slider(
                "Probability of being assigned to the treatment group",
                0.05,
                0.95,
                0.50,
                0.01,
                help="Share of participants who receive treatment assignment (e.g. invitation to training).",
            )
            target_p = st.slider(
                "Participation rate in training among treatment-assigned participants",
                0.00, 1.00, 0.85, 0.01,
                help="Share of assigned participants who actually participate in the training (compliance rate).",
            )

            target_q = st.slider(
                "Participation rate in training among control-assigned participants",
                0.00, 1.00, 0.15, 0.01,
                help="Share of control participants who still participate in the training (self-selection rate).",
            )

            if target_p == 0 and target_q == 0:
                st.warning(
                    "With no participation in either group, the models cannot be estimated. Adjust the participation rates to see results."
                )
                st.stop()
            if target_p == 1 and target_q == 1:
                st.warning(
                    "With full participation in both groups, the models cannot be estimated. Adjust the participation rates to see results."
                )
                st.stop()

            st.subheader("Data generating process")
            true_late = st.slider(
                "True causal effect of participation (thousand USD)",
                -1.00, 3.00, 1.00, 0.05,
                help="The true effect of training participation on earnings in the data generating process. The IV estimator should ideally recover this value (up to sampling variability).",
            )

            base_earn = st.slider(
                "Baseline earnings (without participation in training)",
                4.00,
                8.00,
                6.00,
                0.05,
                help="Earnings when untreated and average motivation.",
            )
            gamma_motiv_train = st.slider(
                "Effect of motivation on participation (probability increase per unit of motivation)",
                0.00, 1.00, 0.60, 0.05,
                help="How much motivation increases the probability of participating in the training. This creates confounding bias for naive comparisons.",
            )
            gamma_motiv_earn = st.slider(
                "Effect of motivation on earnings (thousand USD per unit of motivation)",
                0.00, 5.00, 2.00, 0.25,
                help="How much motivation increases earnings independently of training. This creates confounding bias for naive comparisons.",
            )

            st.subheader("Other settings")
            seed = st.number_input(
                "Random seed",
                min_value=0,
                max_value=1_000_000,
                value=42,
                step=1,
                help="Same seed and same settings reproduces the same synthetic sample.",
            )

            params = {
                "n": n,
                "assignment_rate": assignment_rate,
                "target_p": target_p,
                "target_q": target_q,
                "true_late": true_late,
                "base_earn": base_earn,
                "gamma_motiv_train": gamma_motiv_train,
                "gamma_motiv_earn": gamma_motiv_earn,
                "seed": int(seed),
            }

        download_slot = st.empty()

    return page, params, download_slot


def render_intro_page(df: pd.DataFrame) -> None:
    st.title("Using Instrumental Variables to Estimate Causal Effects in Case of Imperfect Compliance")

    p_hat = df.loc[df["assigned"] == 1, "trained"].mean()
    q_hat = df.loc[df["assigned"] == 0, "trained"].mean()
    fs = p_hat - q_hat

    st.markdown("""
### Introduction

In ideal experiments, we randomly assign treatment to measure its effect.
However, in the real world (and in this simulation), people often don't follow their assignment - 
they self-select into participation or do not participate even if they were assigned to it.
This dashboard explores how the *instrumental variable (IV)* approach can recover the true causal effect even when unobserved factors like participant's motivation contaminate our data.
                
More specifically, we simulate a job training program where:
- Participants are randomly assigned to be offered training (treatment) or not (control).
- Not everyone offered training actually participates, and some in the control group find ways to get trained
- Unobserved motivation affects both the likelihood of participating and the earnings of participants, creating confounding bias in naive comparisons.
- We will see how the IV approach can isolate the causal effect of training on earnings for the "compliers" - those who take the training because they were offered it, but wouldn't have otherwise.

##### 1. Core concepts

To get the most out of this tool, it is helpful to understand the three specific effects we are measuring:

* **Naive OLS:** This simply compares those who participated vs. those who didn't. It is usually "biased" because motivated people might be more likely to both participate *and* earn more regardless of the training.
* **Reduced Form / AITTE (average intent-to-treat effect):** This measures the effect of being *offered* the program. Since the offer (assignment) is randomized, this estimate is "clean" but it doesn't tell us the effect of the actual training.
* **IV / LATE (local average treatment effect):** This uses the random assignment as a "lever" to isolate only the variation in training that is clean and randomized. It estimates the effect specifically for *compliers* - those who took the training because they were assigned to it, but wouldn't have otherwise.

##### 2. How to use the dashboard

Use the sidebar to manipulate the "hidden" reality of the data. You might want to try the following experiments:

* **The "Weak Instrument" Crisis:** Lower the difference between the participation rates in the treatment and control groups. Watch how the IV confidence intervals in the forest plot explode as the instrument loses power.
* **The Selection Bias Test:** Increase the effect of motivation on training and earnings using the sliders. Notice how the **Naive OLS** estimate moves further away from the **True Effect** (the dotted line), while the **IV** estimate remains centered.
* **The Balance Check:** Look at the **Distribution Diagnostics**. Since "Assigned" is randomized, the groups should look almost identical across e.g. Age and Education, even if they differ in their final "Trained" status.

##### 3. Navigation

1. **Data Balance & Diagnostics**: Verify that randomization worked and inspect the "Compliance" (how many people actually followed their assignment).
2. **Regression analysis**: Compare the output of different models and see which one successfully "finds" the true causal effect you set in the sidebar.
""")


def render_diagnostics_page(df: pd.DataFrame, params: dict) -> None:
    st.title("Data balance and diagnostics")

    st.markdown("##### Data generating process")
    st.caption(
        "This graph illustrates the causal structure in the synthetic data. You can trace the IV path visually (green path) from assignment through training to earnings, and see how unobserved motivation (red path) can confound the relationship if no IV is used."
    )
    st.graphviz_chart(build_dgp_graphviz(params), use_container_width=True)

    with st.expander('Under the hood: exact data generation process'):
        st.latex(r"""
\begin{aligned}

\textbf{Unobserved Motivation:} \quad 
M_i &\sim \mathcal{N}(0,1) \\

\textbf{Random Assignment (Instrument):} \quad
Z_i &\sim \text{Bernoulli}(p_{assign}) \\

\\
\textbf{Participation Decision (First Stage):} \\

D_i &= \mathbf{1}
\left[
\alpha_{Z_i}
+ \gamma_{train} M_i
+ \varepsilon_i
> 0
\right], 
\quad
\varepsilon_i \sim \mathcal{N}(0,1) \\

\\
\textbf{Outcome Equation:} \\

Y_i &= \beta_0
+ \tau D_i
+ \gamma_{out} M_i
+ \delta X_i
+ u_i,
\quad
u_i \sim \mathcal{N}(0,\sigma^2)

\\[1em]
\textbf{Where:} \\

Y_i &:\ \text{Earnings outcome for individual } i \\

D_i &:\ \text{Participation (treatment received)} \\

Z_i &:\ \text{Random assignment (instrument)} \\

M_i &:\ \text{Unobserved motivation (confounder)} \\

X_i &:\ \text{Vector of observed control variables} \\

\tau &:\ \text{Causal effect of participation on earnings} \\

\alpha_{Z_i} &:\ \text{Assignment-specific participation intercept} \\

\gamma_{train} &:\ \text{Effect of motivation on participation} \\

\gamma_{out} &:\ \text{Effect of motivation on earnings} \\
                 
\beta_0 &:\ \text{Baseline earnings when untreated and average motivation} \\

\delta &:\ \text{Coefficients on observed controls} \\

u_i, \varepsilon_i &:\ \text{Mean-zero error terms}

\end{aligned}
""")

    st.markdown("##### Balance check table")
    st.caption(
        "This table compares means of background variables across assigned groups, along with t-tests for differences. In a well-randomized sample, we expect no significant differences."
    )
    balance_vars = ["age", "female", "education_years", "prior_experience_months"]
    balance_df = compute_balance_table(df, balance_vars)
    st.dataframe(
        balance_df.style.format(
            {
                "Treatment mean": "{:.3f}",
                "Control mean": "{:.3f}",
                "Difference": "{:.3f}",
                "t-stat": "{:.3f}",
                "p-value": "{:.4f}",
            }
        ),
        use_container_width=True,
    )

    st.markdown("##### Distribution diagnostics")
    st.caption("These plots show the distribution of key covariates by assigned groups. Overlapping distributions indicate good balance, while large shifts may signal random imbalances in the generated sample.")
    st.pyplot(plot_balance_grid(df), use_container_width=True)

    st.markdown("##### Compliance table")
    st.caption(
        "This table shows the cross-tabulation of actual training participation by assigned groups. The difference in training rates between assigned treatment and control groups indicates the strength of the first stage for IV estimation."
    )
    st.dataframe(compliance_table(df), use_container_width=True)


def render_regression_page(df: pd.DataFrame, params: dict) -> None:
    st.title("Regression analysis")
    st.caption(
        "This section compares different estimation strategies. The table summarizes point estimates, standard errors, confidence intervals, and interpretations. The plot visually compares the estimates and their uncertainty to the true effect."
    )

    controls = ["age", "female", "education_years", "prior_experience_months"]
    model_summary, results_df = run_models(df, controls)

    st.markdown("##### Model estimation table")
    st.dataframe(
        results_df.set_index("Model").style.format(
            {
                "Estimate": "{:.4f}",
                "SE": "{:.4f}",
                "p-value": "{:.4f}",
                "95% CI low": "{:.4f}",
                "95% CI high": "{:.4f}",
            }
        ),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Estimator comparison with confidence intervals")
        st.pyplot(plot_estimator_comparison(results_df, params["true_late"]), use_container_width=True)
    with c2:
        st.markdown("##### IV estimation details")
        st.graphviz_chart(
            build_iv_graphviz(
                {
                    "iv_late": model_summary["wald"],
                    "first_stage_coef": model_summary["first_stage_coef"],
                    "aitte": model_summary["aitte"],
                }
            ),
            use_container_width=True,
        )


def main() -> None:
    page, params, download_slot = render_shared_sidebar()

    st.session_state.df = generate_synthetic_data(**params)

    df = st.session_state.df
    download_slot.download_button(
        label="Download synthetic data (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="synthetic_iv_data.csv",
        mime="text/csv",
        use_container_width=True,
        help="Downloads the currently simulated dataset with all participant-level variables.",
    )

    if page == "Introduction":
        render_intro_page(df)
    elif page == "Data balance and diagnostics":
        render_diagnostics_page(df, params)
    else:
        render_regression_page(df, params)


if __name__ == "__main__":
    main()
