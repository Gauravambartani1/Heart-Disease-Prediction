import pickle

import pandas as pd
import streamlit as st
import torch

st.set_page_config(page_title="Heart Disease Risk", page_icon="🫀", layout="wide")

# Risk bands on the predicted probability. 0.35 is the sensitivity-oriented cutoff
# from the notebook; 0.65 mirrors it. Not trained classes: tune them here.
LOW_CUTOFF, HIGH_CUTOFF = 0.35, 0.65
BANDS = {
    "low": ("Low risk", "#2BD4A0", "The model finds few signs of heart disease in these readings."),
    "moderate": ("Moderate risk", "#F5A524",
                 "The readings are mixed. The model can't clearly separate this profile from heart disease, so a clinical check is worthwhile."),
    "high": ("High risk", "#FF5468",
             "The model finds a pattern consistent with heart disease. Prompt clinical follow-up is recommended."),
}


def band_of(p):
    return "low" if p < LOW_CUTOFF else "moderate" if p < HIGH_CUTOFF else "high"


def html(s):
    """Strip indentation and blank lines so Markdown never turns the HTML into a code block."""
    return "\n".join(l.strip() for l in s.splitlines() if l.strip())


# --- STYLES ---
st.markdown(html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500..800&family=Figtree:wght@400;500;600&display=swap');
@property --n { syntax:'<integer>'; inherits:false; initial-value:0; }
:root { --bg:#0A1020; --surf:#111A2E; --line:rgba(255,255,255,.09); --txt:#E8ECF6; --mut:rgba(232,236,246,.62);
        --signal:#FF5468; --amber:#F5A524; --mint:#2BD4A0; }
html, body, [class*="css"], .stApp { font-family:'Figtree',sans-serif; color:var(--txt); }
.stApp { background-color:var(--bg);
  background-image:linear-gradient(rgba(120,150,255,.05) 1px,transparent 1px),linear-gradient(90deg,rgba(120,150,255,.05) 1px,transparent 1px),
                   linear-gradient(rgba(120,150,255,.02) 1px,transparent 1px),linear-gradient(90deg,rgba(120,150,255,.02) 1px,transparent 1px);
  background-size:40px 40px,40px 40px,8px 8px,8px 8px; }
[data-testid="stHeader"] { background:transparent; }
#MainMenu, footer { visibility:hidden; }
.block-container { max-width:1160px; padding-top:2rem; padding-bottom:4rem; }

h1.t { font-family:'Bricolage Grotesque',sans-serif; font-weight:700; font-size:clamp(2rem,4.5vw,3.2rem); letter-spacing:-.025em; line-height:1.05; margin:0 0 .5rem; }
p.s { color:var(--mut); font-size:1.05rem; max-width:56ch; margin:0; line-height:1.55; }
.ecg { width:100%; height:64px; display:block; margin:1rem 0 .6rem; overflow:visible; }
.ecg path { fill:none; stroke:var(--signal); stroke-width:2.5; stroke-linecap:round; stroke-linejoin:round; }
.ecg .ghost { stroke-opacity:.18; }
.ecg .live { stroke-dasharray:1; filter:drop-shadow(0 0 6px rgba(255,84,104,.75)); animation:trace 5.5s cubic-bezier(.5,0,.3,1) infinite; }
@keyframes trace { 0%{stroke-dashoffset:1;opacity:1} 65%{stroke-dashoffset:0;opacity:1} 88%{stroke-dashoffset:0;opacity:1} 100%{stroke-dashoffset:0;opacity:0} }

/* Nav + model picker as pills */
div[role="radiogroup"] { gap:.5rem; }
div[role="radiogroup"] label { background:var(--surf); border:1px solid var(--line); border-radius:999px; padding:.4rem 1.1rem; cursor:pointer; transition:background .25s,border-color .25s; }
div[role="radiogroup"] label:hover { border-color:rgba(255,255,255,.35); }
div[role="radiogroup"] label > div:first-child { display:none; }
div[role="radiogroup"] label:has(input:checked) { background:var(--txt); border-color:var(--txt); }
div[role="radiogroup"] label:has(input:checked) p { color:#0A1020; font-weight:600; }

/* Input card */
[data-testid="stVerticalBlockBorderWrapper"] { background:var(--surf); border:1px solid var(--line) !important; border-radius:18px !important; }
.grp { font-family:'Bricolage Grotesque',sans-serif; font-weight:600; font-size:1.05rem; margin:.3rem 0 .4rem; }
[data-testid="stWidgetLabel"] p { font-weight:500; font-size:.88rem; color:var(--mut); }
div[data-baseweb="select"] > div { border-radius:10px; border-color:var(--line); background:#0D1526; transition:border-color .2s, box-shadow .2s; }
div[data-baseweb="select"] > div:focus-within { border-color:var(--signal); box-shadow:0 0 0 3px rgba(255,84,104,.2); }

/* Result panel */
[data-testid="stColumn"]:has(.anchor), [data-testid="column"]:has(.anchor) { position:sticky; top:1.2rem; align-self:flex-start; }
.res { background:var(--surf); border:1px solid var(--line); border-top:4px solid var(--c); border-radius:18px; padding:1.5rem 1.6rem 1.4rem;
       transition:border-color .5s; box-shadow:0 30px 60px -40px var(--c); }
.res h3 { font-family:'Bricolage Grotesque',sans-serif; font-size:1.9rem; margin:0 0 .2rem; color:var(--c); letter-spacing:-.01em; transition:color .5s; }
.big { font-family:'Bricolage Grotesque',sans-serif; font-weight:700; font-size:4.6rem; line-height:1; letter-spacing:-.03em;
       counter-reset:n var(--n); transition:--n .9s cubic-bezier(.2,.7,.2,1); animation:count 1.2s cubic-bezier(.2,.7,.2,1); }
.big::after { content:counter(n) '%'; }
@keyframes count { from { --n:0; } }
.cap { color:var(--mut); font-size:.85rem; margin:.15rem 0 1rem; }
.scale { position:relative; height:12px; border-radius:99px; display:flex; overflow:visible; margin:1.4rem 0 .4rem; }
.scale i { height:100%; opacity:.9; }
.scale i:nth-child(1){ width:35%; background:var(--mint); border-radius:99px 0 0 99px; }
.scale i:nth-child(2){ width:30%; background:var(--amber); }
.scale i:nth-child(3){ width:35%; background:var(--signal); border-radius:0 99px 99px 0; }
.pin { position:absolute; top:-7px; width:4px; height:26px; margin-left:-2px; background:#fff; border-radius:3px; box-shadow:0 0 0 3px var(--surf);
       left:var(--p); transition:left .8s cubic-bezier(.2,.7,.2,1); animation:pin 1.2s cubic-bezier(.2,.7,.2,1); }
@keyframes pin { from { left:0; } }
.ticks { display:flex; font-size:.78rem; color:var(--mut); }
.ticks span:nth-child(1){ width:35%; } .ticks span:nth-child(2){ width:30%; text-align:center; } .ticks span:nth-child(3){ width:35%; text-align:right; }
.row { display:flex; justify-content:space-between; font-size:.9rem; padding:.5rem 0; border-top:1px solid var(--line); }
.row b { font-weight:600; }
.fine { color:var(--mut); font-size:.78rem; margin-top:.8rem; line-height:1.45; }

/* About */
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:.9rem; margin:1.2rem 0 2rem; }
.tile { background:var(--surf); border:1px solid var(--line); border-radius:16px; padding:1.1rem 1.2rem; }
.tile b { display:block; font-family:'Bricolage Grotesque',sans-serif; font-size:2.1rem; line-height:1.1; }
.tile span { color:var(--mut); font-size:.85rem; }
.about h2 { font-family:'Bricolage Grotesque',sans-serif; font-size:1.5rem; margin:2.2rem 0 .6rem; letter-spacing:-.01em; }
.about p, .about li { color:rgba(232,236,246,.82); line-height:1.65; max-width:72ch; }
.about ul { padding-left:1.1rem; }
.step { display:flex; gap:1rem; padding-bottom:1.3rem; position:relative; }
.step:not(:last-child)::before { content:''; position:absolute; left:15px; top:34px; bottom:2px; width:2px; background:var(--line); }
.step em { flex:none; width:32px; height:32px; border-radius:50%; background:var(--signal); color:#0A1020; font-style:normal; font-weight:700; display:grid; place-items:center; }
.step h4 { margin:.2rem 0 .2rem; font-size:1.02rem; }
.step p { margin:0; font-size:.95rem; }
.tw { overflow-x:auto; }
table.m { border-collapse:collapse; width:100%; font-size:.92rem; }
table.m th, table.m td { text-align:left; padding:.65rem .9rem; border-bottom:1px solid var(--line); white-space:nowrap; }
table.m th { color:var(--mut); font-weight:500; }
table.m tr.on td { color:#fff; font-weight:600; }
.warn { border-left:4px solid var(--amber); background:rgba(245,165,36,.08); padding:.9rem 1.1rem; border-radius:0 12px 12px 0; margin-top:1.2rem; }

@media (prefers-reduced-motion: reduce) { * { animation:none !important; transition:none !important; } }
</style>
"""), unsafe_allow_html=True)


# --- HEARTBEAT TRACE ---
def ecg_path(beats=6, width=1200, base=70):
    bw, d = width / beats, [f"M0,{base}"]
    for b in range(beats):
        x = b * bw
        d += [f"L{x+40:.0f},{base}", f"Q{x+52:.0f},{base-14} {x+64:.0f},{base}", f"L{x+85:.0f},{base}",
              f"L{x+94:.0f},{base+9}", f"L{x+106:.0f},{base-58}", f"L{x+118:.0f},{base+30}", f"L{x+127:.0f},{base}",
              f"L{x+150:.0f},{base}", f"Q{x+170:.0f},{base-26} {x+190:.0f},{base}", f"L{x+bw:.0f},{base}"]
    return " ".join(d)


P = ecg_path()
ECG = f'<svg class="ecg" viewBox="0 0 1200 130" preserveAspectRatio="none" aria-hidden="true"><path class="ghost" d="{P}"/><path class="live" pathLength="1" d="{P}"/></svg>'

# --- HEADER + NAV ---
st.markdown(html(f"""
<h1 class="t">Heart disease risk</h1>
<p class="s">Enter a patient's readings and get an instant risk estimate from two models trained on the UCI heart disease data.</p>
{ECG}
"""), unsafe_allow_html=True)
page = st.radio("Page", ["Assessment", "About the model"], horizontal=True, label_visibility="collapsed", key="page")
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)


# --- ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('log_reg_model.pkl', 'rb') as f:
        log_reg = pickle.load(f)
    with open('expected_columns.pkl', 'rb') as f:
        expected_cols = pickle.load(f)
    mlp_weights = torch.load('mlp_weights.pt')
    return scaler, log_reg, mlp_weights, expected_cols


def predict(inputs, scaler, log_reg, w, expected_cols):
    """Same preprocessing and models as training. Returns (logistic probability, MLP probability)."""
    X = pd.get_dummies(pd.DataFrame([inputs]))
    # Training used drop_first=True, so a baseline category is simply all zeros: missing columns -> 0
    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[expected_cols]
    scale_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    X[scale_cols] = scaler.transform(X[scale_cols])
    X = X.astype(float)

    p_lr = float(log_reg.predict_proba(X)[0][1])
    X_t = torch.tensor(X.values, dtype=torch.float32)
    with torch.no_grad():
        h = torch.tanh(X_t @ w['W1'] + w['b1'])
        p_mlp = torch.sigmoid(h @ w['W2'] + w['b2']).item()
    return p_lr, p_mlp


# ============================ ASSESSMENT ============================
if page == "Assessment":
    scaler, log_reg, mlp_weights, expected_cols = load_artifacts()
    left, right = st.columns([3, 2], gap="large")

    with left:
        with st.container(border=True):
            st.markdown('<div class="grp">Patient</div>', unsafe_allow_html=True)
            a, b = st.columns(2)
            age = a.slider("Age", 20, 100, 54)
            sex = b.selectbox("Sex", ["Male", "Female"])

            st.markdown('<div class="grp">Symptoms</div>', unsafe_allow_html=True)
            a, b = st.columns(2)
            cp = a.selectbox("Chest pain type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
            exang = b.selectbox("Exercise-induced angina?", ["False", "True", "unknown"])

            st.markdown('<div class="grp">Vitals</div>', unsafe_allow_html=True)
            a, b = st.columns(2)
            trestbps = a.slider("Resting blood pressure (mm Hg)", 90, 200, 130)
            chol = b.slider("Cholesterol (mg/dl)", 100, 450, 240)
            thalch = a.slider("Max heart rate achieved", 70, 210, 140)
            fbs = b.selectbox("Fasting blood sugar > 120 mg/dl", ["False", "True", "Unknown"])

            st.markdown('<div class="grp">ECG and stress test</div>', unsafe_allow_html=True)
            a, b = st.columns(2)
            restecg = a.selectbox("Resting ECG", ["normal", "st-t abnormality", "lv hypertrophy"])
            oldpeak = b.slider("ST depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
            thal = a.selectbox("Thallium stress result", ["normal", "fixed defect", "reversable defect", "unknown"])

    # `ca` and `slope` were dropped before training, so they are not inputs.
    inputs = {'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
              'restecg': restecg, 'thalch': thalch, 'exang': exang, 'oldpeak': oldpeak, 'thal': thal}
    p_lr, p_mlp = predict(inputs, scaler, log_reg, mlp_weights, expected_cols)

    with right:
        st.markdown('<div class="anchor"></div>', unsafe_allow_html=True)
        model_choice = st.radio("Model", ["Logistic regression", "Neural network"], horizontal=True, label_visibility="collapsed")
        p = p_lr if model_choice == "Logistic regression" else p_mlp
        head, color, body = BANDS[band_of(p)]
        pct = round(p * 100)
        st.markdown(html(f"""
        <div class="res" style="--c:{color}">
        <h3>{head}</h3>
        <div class="big" style="--n:{pct}" role="img" aria-label="{pct} percent"></div>
        <div class="cap">Predicted probability of heart disease, {model_choice.lower()}</div>
        <p style="margin:0;line-height:1.55">{body}</p>
        <div class="scale"><i></i><i></i><i></i><div class="pin" style="--p:{p*100:.1f}%"></div></div>
        <div class="ticks"><span>Low</span><span>Moderate</span><span>High</span></div>
        <div class="row" style="margin-top:1.2rem"><span>Logistic regression</span><b>{p_lr*100:.0f}%</b></div>
        <div class="row"><span>Neural network</span><b>{p_mlp*100:.0f}%</b></div>
        <div class="fine">Decision support, not a diagnosis. A clinician should review every result.
        Bands: low under {LOW_CUTOFF:.0%}, moderate {LOW_CUTOFF:.0%} to {HIGH_CUTOFF:.0%}, high above.</div>
        </div>
        """), unsafe_allow_html=True)

# ============================== ABOUT ===============================
else:
    steps = [
        ("Data", "UCI Heart Disease data: patients from four hospitals (Cleveland, Hungary, Switzerland, VA Long Beach), loaded from a Kaggle copy. Each row has 13 clinical measurements and a severity label from 0 to 4."),
        ("Target", "Severity 1 to 4 were merged into one class, so the label is binary: heart disease or none. Both models predict that probability."),
        ("Cleaning", "Blood pressure and cholesterol readings of 0 are impossible, so they were treated as missing. Missing blood pressure was filled with the average for the patient's 10-year age band and sex. Missing max heart rate was filled with the age-band average. Cholesterol and ST depression were filled with an iterative imputer using ExtraTrees. Missing fasting blood sugar, exercise angina and thallium results became their own &quot;unknown&quot; category. Two rows without a resting ECG were dropped, leaving 918 patients."),
        ("Features", "<code>ca</code> (vessels colored) and <code>slope</code> (ST slope) were dropped because too many values were missing. Categories were one-hot encoded, and five continuous measures (age, blood pressure, cholesterol, max heart rate, ST depression) were standardized. That gives 18 input features."),
        ("Split", "80% training and 20% testing, stratified on the label with a fixed seed: 734 patients to train, 184 to test. The scaler was fit on the training set only."),
        ("Logistic regression", "A grid search with 5-fold cross-validation, scored on ROC-AUC, tried 6 regularization strengths with L1 and L2 penalties. The best was L2 with C = 10 (cross-validated AUC 0.881)."),
        ("Neural network", "Written from scratch in PyTorch: 18 inputs, one hidden layer of 100 tanh units, one sigmoid output. Weights were initialized for tanh, and the model was trained with binary cross-entropy and plain gradient descent (learning rate 0.01, 2,100 full-batch steps). Training loss fell from 0.656 to 0.404."),
    ]
    steps_html = "".join(f'<div class="step"><em>{i}</em><div><h4>{t}</h4><p>{d}</p></div></div>' for i, (t, d) in enumerate(steps, 1))
    st.markdown(html(f"""
    <div class="about">
    <div class="tiles">
    <div class="tile"><b>0.93</b><span>Test ROC-AUC, logistic regression</span></div>
    <div class="tile"><b>85%</b><span>Test accuracy, both models</span></div>
    <div class="tile"><b>93%</b><span>Of patients with disease caught</span></div>
    <div class="tile"><b>918</b><span>Patients (734 train, 184 test)</span></div>
    </div>
    <h2>How the models were built</h2>
    {steps_html}
    <h2>Results on the held-out test set</h2>
    <div class="tw"><table class="m">
    <tr><th>Model</th><th>ROC-AUC</th><th>Accuracy</th><th>Sensitivity</th><th>Specificity</th></tr>
    <tr class="on"><td>Logistic regression, cutoff 0.50</td><td>0.931</td><td>85%</td><td>93%</td><td>76%</td></tr>
    <tr><td>Logistic regression, cutoff 0.35</td><td>0.931</td><td>82%</td><td>94%</td><td>66%</td></tr>
    <tr class="on"><td>Neural network, cutoff 0.50</td><td>not measured</td><td>85%</td><td>93%</td><td>76%</td></tr>
    <tr><td>Random forest (tested, not used here)</td><td>0.926</td><td>81%</td><td>84%</td><td>77%</td></tr>
    </table></div>
    <p style="margin-top:.8rem">Sensitivity is the share of patients with disease the model flags. Specificity is the share of healthy patients it clears.</p>
    <h2>Where Moderate risk comes from</h2>
    <p>The models only learned two classes, so Moderate is not a trained label. It is a band of uncertainty on the predicted probability. In the notebook, lowering the cutoff to 0.35 caught slightly more disease but cleared far fewer healthy patients (specificity fell from 76% to 66%). So scores under 35% are Low, and scores of 65% or more mirror that cutoff as High. Everything between is Moderate. These cutoffs are a design choice, and the probabilities are not calibrated.</p>
    <h2>Limits to keep in mind</h2>
    <ul>
    <li>The dataset is small (918 patients) and comes from four hospitals. The models have not been checked on any outside population.</li>
    <li>The imputers were fit on all rows before the train/test split, so test scores may be slightly optimistic.</li>
    <li>The test set is 184 patients, so the metrics above can shift by several points with a different split.</li>
    </ul>
    <div class="warn">This tool is for exploration and learning. It is not a medical device and must not replace a clinician's judgment.</div>
    </div>
    """), unsafe_allow_html=True)