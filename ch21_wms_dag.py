import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.formula.api as smf

color = ["#3a5e8c", "#10a53d", "#541352", "#ffcf20", "#2f9aa0"]

st.set_page_config(layout="wide", page_title="Ch. 21 - WMS DAG")
st.title("Ch. 21 - Exploring Confounders and Bad Conditioners in the WMS dataset")

st.markdown("""
### Introduction

This dashboard is designed to help explore **causal inference concepts** using the *World Management Survey (WMS)* dataset.  
The central question is whether **family/founder ownership** (X) has a causal impact on the **management score** (Y).  
To address this, we need to carefully consider which variables to condition on when estimating causal effects.

### Key Functionalities

- **Assign Roles to Variables:**  
  In the sidebar, you can specify the causal role of each variable (e.g., *Common Cause*, *Mechanism*, *Collider*, etc.).  
  These choices determine which variables are included in the regression model and how the DAG (Directed Acyclic Graph) is drawn.

- **Conditioning Options:**  
  Variables classified as *Common Causes*, *Reverse Causality*, or *Unwanted Mechanisms* are automatically added to the regression model.  
  You can also toggle whether to use more appropriate functional forms (e.g., log transformations, squared terms).

- **Visual Causal Diagram (DAG):**  
  The DAG updates based on your inputs, showing which variables are conditioned on (yellow), which are not (gray), and highlighting the treatment (blue) and outcome (green).

- **Regression Results Table:**  
  The dashboard compares the **Base Model** (regressing management score on ownership only) with your **Constructed Model** (ownership plus selected conditioning variables).  
  Coefficients, standard errors, significance levels, and R-squared are displayed for comparison.

### Things to Observe

- How does the estimated effect of **ownership on management** change as you condition on different variables?  
- Do some variables introduce bias when conditioned on (e.g., colliders)?  
- Does including *mechanisms* block part of the causal effect you want to capture?  
- Compare the **Base Model** vs. **Constructed Model** to see how careful variable selection matters for causal inference.

""")


st.sidebar.header("Settings")
st.sidebar.subheader("Role of Variables in DAG")

role_options = ["Common Cause", "Reverse Causality", "Unwanted Mechanism",
                "Mechanism", "Collider", "Exog. Variation in X",
                "Exog. Variation in Y"]
roles_to_condition = role_options[:3]
roles_not_to_condition = role_options[3:]

# Mapping variable names to pretty labels
label_map = {
    'degree_m': "Share of managers with college degree",
    'degree_nm': "Share of non-managers with college degree",
    'export': "Share of production exported",
    'emp_firm': "Number of employees",
    'compet_cat': "Competition",
    'industry': "Industry",
    'countrycode': "Country",
    'age_cat': "Firm age"
}

# Default roles
default_roles = {
    'degree_m': "Mechanism",
    'degree_nm': "Common Cause",
    'export': "Collider",
    'emp_firm': "Reverse Causality",
    'compet_cat': "Common Cause",
    'industry': "Common Cause",
    'countrycode': "Common Cause",
    'age_cat': "Reverse Causality"
}

roles = {}

# Table header
c1, c2 = st.sidebar.columns([2, 3])
c1.markdown("**Variable**")
c2.markdown("**Role**")

# Build rows with preserved defaults
for var, label in label_map.items():
    c1, c2 = st.sidebar.columns([2, 3])
    c1.markdown(label)
    roles[var] = c2.selectbox(
        "",
        options=role_options,
        index=role_options.index(default_roles[var]),  # preserve default
        key=f"role_{var}"
    )


st.sidebar.subheader("Other Settings")
funcform = st.sidebar.toggle("Use proper functional form of variables", value=True, key='funcform')

@st.cache_data
def load_data():
    data = pd.read_csv("data/wms_da_textbook-work.csv")
    return data

data = load_data()

st.sidebar.download_button(
    label="Download Data",
    data=data.to_csv(index=False),
    file_name="ch21_wms_data.csv",
    mime="text/csv"
)

vars_to_condition = [var for var, role in roles.items() if role in roles_to_condition]
if funcform:
    if 'degree_m' in vars_to_condition:
        vars_to_condition.append('degree_m_sq')
    if 'degree_nm' in vars_to_condition:
        vars_to_condition.append('degree_nm_sq')
    if 'emp_firm' in vars_to_condition:
        vars_to_condition.append('lnemp')
        vars_to_condition.remove('emp_firm')

base_model = smf.ols("management ~ foundfam_owned", data=data).fit(cov_type='HC1')
user_model = smf.ols("management ~ foundfam_owned + " + " + ".join(vars_to_condition), data=data).fit(cov_type='HC1')

def add_stars(val):
    if val < 0.01:
        return "***"
    elif val < 0.05:
        return "**"
    elif val < 0.1:
        return "*"
    else:
        return ""

results_dict = {
    'Base Model': {
        'Coefficient': f"{base_model.params['foundfam_owned']:.3f}{add_stars(base_model.pvalues['foundfam_owned'])}",
        'SE' : f"({base_model.bse['foundfam_owned']:.3f})",
        'R-squared': f"{base_model.rsquared:.3f}"
    },
    'Constructed Model': {
        'Coefficient': f"{user_model.params['foundfam_owned']:.3f}{add_stars(user_model.pvalues['foundfam_owned'])}",
        'SE' : f"({user_model.bse['foundfam_owned']:.3f})",
        'R-squared': f"{user_model.rsquared:.3f}"
    }
}

results_df = pd.DataFrame(results_dict).T

def make_dag_string(roles, vars_to_condition):
    if 'lnemp' in vars_to_condition:
        vars_to_condition.append('emp_firm')
        vars_to_condition.remove('lnemp')

    # Map variable names to readable labels
    label_map = {
        "degree_m": "Mgr College Share",
        "degree_nm": "Non-Mgr College Share",
        "export": "Export Share",
        "emp_firm": "Employment",
        "compet_cat": "Competition",
        "industry": "Industry",
        "countrycode": "Country",
        "age_cat": "Firm Age",
    }

    # Helper for coloring
    def node_style(var, is_x=False, is_y=False):
        if is_x:
            return 'style="filled,rounded", fillcolor="#3a5e8c", fontcolor="white"'
        if is_y:
            return 'style="filled,rounded", fillcolor="#10a53d", fontcolor="white"'
        if var in vars_to_condition:
            return 'style="filled,rounded", fillcolor="#ffcf20"'
        else:
            return 'style="filled,rounded", fillcolor="lightgray"'

    # Build node strings by role group
    role_nodes = {r: [] for r in set(roles.values())}

    for var, role in roles.items():
        if role is None:
            continue
        label = label_map.get(var, var)
        role_nodes[role].append(f'{var} [label="{label}", {node_style(var)}];')

    # Graphviz string
    dot = f"""
digraph G {{
  rankdir=TB;
  node [shape=box, style=rounded, fontsize=12];

  // Always present X and Y
  X [label="Family/Founder Ownership", {node_style("X", is_x=True)}];
  Y [label="Management Score", {node_style("Y", is_y=True)}];

  // Nodes by quadrant
  // Top left: common cause confounders
  {"".join(role_nodes.get("Common Cause", []))}

  // Top middle: X (already defined)

  // Top right: exogenous X sources
  {"".join(role_nodes.get("Exog. Variation in X", []))}

  // Middle left: reverse causality
  {"".join(role_nodes.get("Reverse Causality", []))}

  // Middle center: mechanisms (direct + unwanted)
  {"".join(role_nodes.get("Mechanism", []))}
  {"".join(role_nodes.get("Unwanted Mechanism", []))}

  // Middle right: colliders
  {"".join(role_nodes.get("Collider", []))}

  // Bottom left: exogenous Y sources
  {"".join(role_nodes.get("Exog. Variation in Y", []))}

  // Bottom middle: Y (already defined)

  // Invisible edges to keep alignment
  {{ rank=same; X }}
  {{ rank=same; {" ".join(role_nodes.get("Common Cause", []))} }}
  {{ rank=same; {" ".join(role_nodes.get("Exog. Variation in X", []))} }}
  {{ rank=same; {" ".join(role_nodes.get("Reverse Causality", []))} }}
  {{ rank=same; {" ".join(role_nodes.get("Mechanism", []))} {" ".join(role_nodes.get("Unwanted Mechanism", []))} }}
  {{ rank=same; {" ".join(role_nodes.get("Collider", []))} }}
  {{ rank=same; {" ".join(role_nodes.get("Exog. Variation in Y", []))} }}
  {{ rank=same; Y }}

  // Invisible alignment helpers
  X -> Y [style=invis];

  // Real causal edges
  X -> Y;

"""

    # Add real edges for each role
    for var, role in roles.items():
        if role is None:
            continue
        if role == "Common Cause":
            dot += f"{var} -> X;\n{var} -> Y;\n"
        elif role == "Mechanism":
            dot += f"X -> {var};\n{var} -> Y;\n"
        elif role == "Unwanted Mechanism":
            dot += f"X -> {var};\n{var} -> Y;\n"
        elif role == "Reverse Causality":
            dot += f"Y -> {var};\n{var} -> X;\n"
        elif role == "Collider":
            dot += f"X -> {var};\nY -> {var};\n"
        elif role == "Exog. Variation in X":
            dot += f"{var} -> X;\n"
        elif role == "Exog. Variation in Y":
            dot += f"{var} -> Y;\n"

    dot += "}\n"
    return dot
dot = make_dag_string(roles, vars_to_condition)

def show_legend():
    fig, ax = plt.subplots(figsize=(1, 0.2))
    ax.axis("off")

    patches = [
        mpatches.Patch(color="#3a5e8c", label="Treatment"),
        mpatches.Patch(color="#10a53d", label="Outcome"),
        mpatches.Patch(color="#ffcf20", label="Conditioned Variable"),
        mpatches.Patch(color="lightgray", label="Not Conditioned Variable"),
    ]

    ax.legend(
        handles=patches,
        loc="center",
        ncol=4,
        frameon=False,
        fontsize=5,
        handlelength=1.5,
        handleheight=1.5
    )

    st.pyplot(fig, clear_figure=True, use_container_width=False)

st.subheader("Causal Diagram (DAG) Based on Inputs")
show_legend()
st.graphviz_chart(dot)

st.subheader("Regression Results")
st.dataframe(results_df, use_container_width=True)