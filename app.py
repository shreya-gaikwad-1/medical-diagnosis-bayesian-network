import pandas as pd
import streamlit as st

import medical_diagnosis_bn as bn

st.set_page_config(page_title="Medical Diagnosis BN", page_icon="🩺", layout="wide")
st.title("🩺 Medical Diagnosis with Bayesian Networks")
st.caption("Advanced AI mini project: exact inference, sampling, and decision theory")

# ---------------- Sidebar: evidence + utilities ----------------
st.sidebar.header("1. Patient evidence")
CHOICES = {"Unknown": None, "Yes": 1, "No": 0}
evidence = {}
for var in ["Smoker", "Pollution", "XRay", "Dyspnea"]:
    label = {"XRay": "X-Ray positive?", "Dyspnea": "Has Dyspnea?",
             "Pollution": "High pollution area?", "Smoker": "Smoker?"}[var]
    val = CHOICES[st.sidebar.radio(label, list(CHOICES), horizontal=True, key=var)]
    if val is not None:
        evidence[var] = val

st.sidebar.header("2. Utilities (payoffs)")
bn.UTILITY[("Treat", 1)] = st.sidebar.slider("Treat, has cancer", -200, 200, 80)
bn.UTILITY[("Treat", 0)] = st.sidebar.slider("Treat, no cancer", -200, 200, -20)
bn.UTILITY[("Don't treat", 1)] = st.sidebar.slider("Don't treat, has cancer", -200, 200, -100)
bn.UTILITY[("Don't treat", 0)] = st.sidebar.slider("Don't treat, no cancer", -200, 200, 0)

n_samples = st.sidebar.select_slider("Samples for approximation",
                                     [100, 1000, 10000, 100000], value=10000)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Diagnosis", "⚖️ Decision", "🎲 Exact vs Sampling", "🕸️ Network"])

# ---------------- Tab 1: diagnosis ----------------
with tab1:
    exact = bn.variable_elimination("Cancer", evidence)
    c1, c2 = st.columns([1, 2])
    c1.metric("P(Cancer)", f"{exact[1]:.2%}")
    c1.write(f"**Evidence given:** {evidence or 'none'}")
    c2.bar_chart(pd.DataFrame({"Probability": [exact[0], exact[1]]},
                              index=["No cancer", "Cancer"]))
    st.subheader("How each finding shifts the belief")
    rows, running = [("Prior (no evidence)", bn.variable_elimination("Cancer", {})[1])], {}
    for k, v in evidence.items():
        running[k] = v
        rows.append((f"+ {k} = {'Yes' if v else 'No'}",
                     bn.variable_elimination("Cancer", running)[1]))
    df = pd.DataFrame(rows, columns=["Step", "P(Cancer)"]).set_index("Step")
    st.line_chart(df)

# ---------------- Tab 2: decision ----------------
with tab2:
    action, eu, eus = bn.best_action(evidence)
    st.success(f"Recommended action: **{action}** (expected utility {eu:.2f})")
    st.bar_chart(pd.DataFrame({"Expected utility": eus}))
    st.subheader("Is it worth ordering another test?")
    cols = st.columns(2)
    for col, test in zip(cols, ["XRay", "Dyspnea"]):
        if test in evidence:
            col.info(f"{test} is already observed.")
        else:
            col.metric(f"Value of information: {test}",
                       f"{bn.value_of_information(test, evidence):.2f}")
    st.caption("VOI = how much the expected utility improves if you observe the "
               "test before deciding. Zero means the test would not change your decision.")

# ---------------- Tab 3: exact vs sampling ----------------
with tab3:
    approx = bn.likelihood_weighting("Cancer", evidence, n=n_samples)
    a, b, c = st.columns(3)
    a.metric("Exact (Variable Elimination)", f"{exact[1]:.4f}")
    b.metric(f"Likelihood Weighting ({n_samples:,})", f"{approx[1]:.4f}")
    c.metric("Absolute error", f"{abs(exact[1] - approx[1]):.4f}")
    st.subheader("Convergence as samples increase")
    sizes = [100, 300, 1000, 3000, 10000, 30000]
    errs = [abs(bn.likelihood_weighting("Cancer", evidence, n=n, seed=1)[1] - exact[1])
            for n in sizes]
    st.line_chart(pd.DataFrame({"Error": errs}, index=sizes))

# ---------------- Tab 4: network ----------------
with tab4:
    dot = ["digraph G { rankdir=LR; node [shape=ellipse, style=filled, fillcolor=lightblue];"]
    for var, (parents, _) in bn.BN.items():
        if var in evidence:
            dot.append(f'{var} [fillcolor=lightgreen, label="{var}\\n= {evidence[var]}"];')
        for p in parents:
            dot.append(f"{p} -> {var};")
    dot.append("}")
    st.graphviz_chart("\n".join(dot))
    st.caption("Green nodes are observed evidence.")
    st.subheader("Conditional probability tables")
    for var, (parents, cpt) in bn.BN.items():
        t = pd.DataFrame([{**dict(zip(parents, k)), f"P({var}=1)": v}
                          for k, v in cpt.items()])
        st.write(f"**{var}**")
        st.dataframe(t, hide_index=True)