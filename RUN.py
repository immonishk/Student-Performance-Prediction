
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "student_performance.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "study_hours_per_day",
    "attendance_percentage",
    "previous_exam_score",
    "assignment_average",
    "sleep_hours_per_night",
    "extracurricular_hours_per_week",
]
TARGET = "final_score"


def train_models():
    """Train the AMML and NNDL models and save their results."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Dataset not found. Make sure data/student_performance.csv is present."
        )

    # Load dataset and clean column names.
    data = pd.read_csv(DATA_FILE)
    data.columns = data.columns.str.strip()

    # Check that all required columns are present.
    missing = set(FEATURES + [TARGET]) - set(data.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing)}"
        )

    # Clean numeric values such as "56. 8" and convert to numbers.
    for col in FEATURES + [TARGET]:
        cleaned = (
            data[col]
            .astype("string")
            .str.strip()
            .str.replace(
                r"(?<=\d)\s*\.\s*(?=\d)",
                ".",
                regex=True
            )
        )
        data[col] = pd.to_numeric(cleaned, errors="coerce")

    # Remove rows with missing or invalid numeric values.
    data = data.dropna(subset=FEATURES + [TARGET]).copy()

    if len(data) < 10:
        raise ValueError(
            "The dataset has fewer than 10 valid rows after cleaning. "
            "Please check data/student_performance.csv."
        )

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Define the four models.
    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),

        "Decision Tree": DecisionTreeRegressor(
            max_depth=6,
            min_samples_leaf=8,
            random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ),

        "ANN (MLP)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=700,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42
            ))
        ])
    }

    rows = []
    trained_models = {}
    predictions = {}

    # Train each model and evaluate its predictions.
    for name, model in models.items():
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)

        trained_models[name] = model
        predictions[name] = predicted

        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, predicted),
            "RMSE": np.sqrt(
                mean_squared_error(y_test, predicted)
            ),
            "R2": r2_score(y_test, predicted)
        })

    metrics = pd.DataFrame(rows).sort_values("RMSE")
    metrics.to_csv(
        OUTPUT_DIR / "model_metrics.csv",
        index=False
    )

    # Save trained models.
    for name, model in trained_models.items():
        slug = (
            name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )
        joblib.dump(
            model,
            OUTPUT_DIR / f"{slug}.joblib"
        )

    # Plot 1: Compare model RMSE values.
    plt.figure(figsize=(8, 4.5))
    plt.bar(metrics["Model"], metrics["RMSE"])
    plt.ylabel("RMSE (lower is better)")
    plt.title("Model comparison")
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "model_rmse_comparison.png",
        dpi=150
    )
    plt.close()

    # Plot 2: ANN training loss.
    ann = trained_models["ANN (MLP)"].named_steps["model"]

    plt.figure(figsize=(7, 4))
    plt.plot(ann.loss_curve_)
    plt.xlabel("Training iteration")
    plt.ylabel("Loss")
    plt.title("ANN training loss")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "ann_training_loss.png",
        dpi=150
    )
    plt.close()

    # Plot 3: Actual versus predicted scores.
    best_name = metrics.iloc[0]["Model"]

    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(
        y_test,
        predictions[best_name],
        alpha=0.65
    )
    plt.plot(
        [0, 100],
        [0, 100],
        linestyle="--"
    )
    plt.xlabel("Actual score")
    plt.ylabel("Predicted score")
    plt.title(f"Actual vs predicted: {best_name}")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "actual_vs_predicted.png",
        dpi=150
    )
    plt.close()

    # Save a summary of the training run.
    summary = {
        "dataset_rows": int(len(data)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "features": FEATURES,
        "target": TARGET,
        "best_model_by_rmse": str(best_name),
        "dataset_type": (
            "synthetic educational demonstration data"
        ),
        "metrics": metrics.to_dict(orient="records")
    }

    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    return trained_models, metrics



@st.cache_resource
def get_models_and_metrics():
    return train_models()



# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performanc Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- AURORA EDITORIAL THEME ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --canvas: #F4F3EC;
    --surface: #FFFFFF;
    --forest: #173C35;
    --forest-2: #245348;
    --lime: #D5F66B;
    --muted: #758178;
    --line: #E1E5DC;
    --soft-green: #E8EFE5;
}

.stApp {
    background: var(--canvas);
    color: var(--forest);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: #173C35;
    border-right: 0;
}

[data-testid="stSidebar"] * {
    color: #F4F6EF;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #245348;
    border-color: #507366;
    border-radius: 10px;
}

.block-container {
    max-width: 1440px;
    padding-top: 2.2rem;
    padding-bottom: 2rem;
}

h1, h2, h3, h4 {
    font-family: 'Manrope', sans-serif !important;
    color: var(--forest) !important;
    letter-spacing: -0.7px;
}

p, label, [data-testid="stMarkdownContainer"] {
    font-family: 'DM Sans', sans-serif;
}

.hero {
    background: var(--forest);
    border-radius: 22px;
    padding: 32px 34px;
    color: #FFFFFF;
    position: relative;
    overflow: hidden;
    margin-bottom: 20px;
    border: 1px solid #2E5C4E;
}

.hero:after {
    content: "";
    position: absolute;
    width: 230px;
    height: 230px;
    right: -75px;
    top: -110px;
    border: 1px solid rgba(213,246,107,0.30);
    border-radius: 50%;
    box-shadow: 0 0 0 25px rgba(213,246,107,0.04),
                0 0 0 55px rgba(213,246,107,0.03);
}

.eyebrow {
    color: #D5F66B;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2.5px;
    margin-bottom: 12px;
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: clamp(30px, 4vw, 45px);
    line-height: 1.13;
    margin: 0 0 12px 0;
    font-weight: 800;
}

.hero p {
    color: #D0DED4;
    font-size: 14px;
    line-height: 1.7;
    max-width: 690px;
    margin: 0;
}

.section-kicker {
    font-size: 11px;
    color: #60806B;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 5px;
}

.panel {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 17px;
    padding: 20px 22px;
    margin: 8px 0 14px 0;
}

.panel h3 {
    margin-top: 5px;
    margin-bottom: 4px;
    font-size: 19px;
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 19px;
    min-height: 118px;
    box-shadow: 0 3px 12px rgba(23,60,53,0.025);
}

.metric-title {
    color: var(--muted);
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 13px;
}

.metric-value {
    color: var(--forest);
    font-family: 'Manrope', sans-serif;
    font-size: 29px;
    line-height: 1.1;
    font-weight: 800;
    letter-spacing: -1px;
}

.metric-note {
    color: #729077;
    font-size: 11px;
    margin-top: 9px;
}

.result-card {
    background: var(--forest);
    border: 1px solid #2E5C4E;
    border-radius: 20px;
    padding: 25px;
    color: #FFFFFF;
}

.result-label {
    color: #D5F66B;
    font-size: 11px;
    letter-spacing: 2px;
    font-weight: 700;
}

.result-score {
    font-family: 'Manrope', sans-serif;
    font-size: 53px;
    font-weight: 800;
    letter-spacing: -2px;
    color: #FFFFFF;
    margin: 9px 0;
}

.result-score span {
    color: #B5C9B9;
    font-size: 19px;
    letter-spacing: 0;
}

div.stButton > button {
    background: var(--forest);
    color: #FFFFFF;
    border: 1px solid var(--forest);
    border-radius: 11px;
    min-height: 48px;
    font-weight: 700;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    background: #28594B;
    color: #D5F66B;
    border-color: #28594B;
}

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 16px;
}

div[data-testid="stMetric"] label {
    color: var(--muted);
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: var(--forest);
    border-color: var(--forest);
}

hr {
    border-color: var(--line);
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;
}

div[data-testid="stImage"] img {
    border-radius: 14px;
}

.footer {
    border-top: 1px solid var(--line);
    text-align: center;
    color: #879188;
    padding: 22px 0 5px;
    font-size: 11px;
    letter-spacing: 0.6px;
    line-height: 1.8;
}

#MainMenu, footer {
    visibility: hidden;
}

@media (max-width: 700px) {
    .hero {
        padding: 23px 20px;
    }
    .metric-value {
        font-size: 24px;
    }
    .result-score {
        font-size: 43px;
    }
}
/* Main dashboard text: black */
.stApp,
.stApp p,
.stApp label,
.stApp .stMarkdown,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stCaptionContainer"] {
    color: #111827;
}

/* Force primary button text to be white across all states */
.stApp button[kind="primary"],
.stApp button[kind="primary"] p,
.stApp button[kind="primary"]:focus,
.stApp button[kind="primary"]:hover,
.stApp button[kind="primary"]:active {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* Main headings */
h1, h2, h3, h4, h5, h6 {
    color: #000000 !important;
}

/* Model performance cards */
.metric-title {
    color: #111827 !important;
}

.metric-value {
    color: #000000 !important;
}

.metric-note {
    color: #374151 !important;
}

/* Sidebar text */
[data-testid="stSidebar"] {
    background: #173C35;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #FFFFFF !important;
}

/* Keep the hero and prediction result text white */
.hero,
.hero p,
.hero h1,
.result-card,
.result-card div {
    color: #FFFFFF;
}

.hero .eyebrow,
.result-label {
    color: #D5F66B;
}

/* White cards with black text */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E1E5DC;
}

div[data-testid="stMetric"] {
    background: #FFFFFF;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #000000 !important;
}

/* Black text in model comparison table */
div[data-testid="stDataFrame"] {
    color: #000000;
}

/* Captions and footer */
.stCaption,
.footer {
    color: #374151 !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>Student Performance<br>Analytics.</h1>
    <p>
        A thoughtful look at academic patterns, study habits and
        machine-learning predictions — designed to turn student
        data into clear, understandable insights.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
try:
    models, metrics = get_models_and_metrics()
except Exception as error:
    st.error(f"Could not load or train the models: {error}")
    st.info(
        "Check data/student_performance.csv, its column names, "
        "and the numeric values in the dataset."
    )
    st.stop()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 8px 0 18px 0;">
            <h2 style="color: #111827; margin-bottom: 4px;">
                Student Performance Analytics
            </h2>
            <p style="color: #EAECEB; font-size: 13px;">Academic Prediction Dashboard</p>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### Prediction settings")

    model_name = st.selectbox(
        "Choose prediction model",
        list(models.keys())
    )

    st.divider()

    st.markdown("### Project overview")

    st.markdown(
        """

        Regression and model evaluation

        Artificial neural network

        **Dataset**

        Synthetic educational records
        """
    )

# ---------------- STUDENT PROFILE ----------------
st.markdown(
    '<div class="section-kicker">01 / INPUT WORKSPACE</div>',
    unsafe_allow_html=True
)
st.header("Build a student profile")
st.write(
    "Adjust the six variables below and explore how the selected "
    "model estimates the final score."
)

academic, habits = st.columns(2, gap="large")

with academic:
    st.markdown("""
    <div class="panel">
        <div class="section-kicker">ACADEMIC INDICATORS</div>
        <h3>📚 Academic profile</h3>
        <p style="color:#758178;font-size:13px">
            Previous results, study time and class participation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    study = st.slider(
        "Daily study hours", 0.0, 9.0, 3.5, 0.5
    )
    attendance = st.slider(
        "Attendance (%)", 40.0, 100.0, 82.0, 1.0
    )
    previous = st.slider(
        "Previous exam score", 0.0, 100.0, 68.0, 1.0
    )

with habits:
    st.markdown("""
    <div class="panel">
        <div class="section-kicker">PERSONAL STUDY ROUTINE</div>
        <h3>🌙 Learning habits</h3>
        <p style="color:#758178;font-size:13px">
            Assignment performance, sleep and outside activities.
        </p>
    </div>
    """, unsafe_allow_html=True)

    assignments = st.slider(
        "Assignment average", 0.0, 100.0, 74.0, 1.0
    )
    sleep = st.slider(
        "Sleep per night (hours)", 3.5, 10.0, 7.0, 0.5
    )
    extra = st.slider(
        "Extracurricular hours per week", 0.0, 6.0, 1.5, 0.5
    )

# ---------------- PROFILE SUMMARY ----------------
st.markdown("---")
st.markdown(
    '<div class="section-kicker">02 / PROFILE SNAPSHOT</div>',
    unsafe_allow_html=True
)
st.subheader("Your current inputs")

p1, p2, p3, p4 = st.columns(4)

profile_cards = [
    ("Daily study", f"{study:.1f} h", "Hours of study per day"),
    ("Attendance", f"{attendance:.0f}%", "Class participation"),
    ("Previous score", f"{previous:.0f}/100", "Prior examination"),
    ("Assignments", f"{assignments:.0f}/100", "Assignment average"),
]

for col, (title, value, note) in zip(
    [p1, p2, p3, p4], profile_cards
):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
st.markdown("---")
st.markdown(
    '<div class="section-kicker">03 / MODEL INFERENCE</div>',
    unsafe_allow_html=True
)
st.subheader("Estimate final performance")
st.write(f"Prediction engine: **{model_name}**")

if st.button("Calculate estimated score", type="primary"):
    sample = pd.DataFrame([{
        "study_hours_per_day": study,
        "attendance_percentage": attendance,
        "previous_exam_score": previous,
        "assignment_average": assignments,
        "sleep_hours_per_night": sleep,
        "extracurricular_hours_per_week": extra
    }])

    score = float(models[model_name].predict(sample)[0])
    score = max(0.0, min(100.0, score))

    result_col, detail_col = st.columns([1, 1.5], gap="large")

    with result_col:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">ESTIMATED FINAL SCORE</div>
            <div class="result-score">
                {score:.1f}<span> / 100</span>
            </div>
            <div style="color:#D0DED4;font-size:13px">
                Generated by {model_name}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with detail_col:
        st.markdown("#### Prediction overview")
        st.progress(int(round(score)))
        st.write(f"**Selected model:** {model_name}")
        st.write(f"**Study time:** {study:.1f} hours/day")
        st.write(f"**Attendance:** {attendance:.0f}%")
        st.write(f"**Previous score:** {previous:.0f}/100")
        st.caption(
            "This is an estimate from a model trained on synthetic data. "
            "It is not a guaranteed result or a measure of student ability."
        )

# ---------------- MODEL PERFORMANCE ----------------

st.markdown(
    """
    <div style="margin: 25px 0 8px 0;">
        <h2 style="color: #111827; margin-bottom: 5px;">
            Model Performance
        </h2>
        <p style="color: #4B5563;">
            Compare the trained regression models using the same
            held-out test dataset.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

best_row = metrics.sort_values("RMSE").iloc[0]

m1, m2, m3 = st.columns(3, gap="medium")

with m1:
    st.markdown(
        f"""
        <div style="
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 20px;
            min-height: 125px;
        ">
            <div style="color: #4B5563; font-size: 14px;">
                Models evaluated
            </div>
            <div style="
                color: #111827;
                font-size: 32px;
                font-weight: 700;
                margin-top: 8px;
            ">{len(metrics)}</div>
            <div style="color: #6B7280; font-size: 12px;">
                Models trained and tested
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m2:
    st.markdown(
        f"""
        <div style="
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 20px;
            min-height: 125px;
        ">
            <div style="color: #4B5563; font-size: 14px;">
                Lowest test RMSE
            </div>
            <div style="
                color: #111827;
                font-size: 32px;
                font-weight: 700;
                margin-top: 8px;
            ">{best_row['RMSE']:.2f}</div>
            <div style="color: #6B7280; font-size: 12px;">
                Lower values indicate smaller errors
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m3:
    st.markdown(
        f"""
        <div style="
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 20px;
            min-height: 125px;
        ">
            <div style="color: #4B5563; font-size: 14px;">
                Best RMSE model
            </div>
            <div style="
                color: #111827;
                font-size: 23px;
                font-weight: 700;
                margin-top: 12px;
            ">{best_row['Model']}</div>
            <div style="color: #6B7280; font-size: 12px;">
                Ranked by test RMSE
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
    <h3 style="color: #111827; margin-top: 28px;">
        Model comparison table
    </h3>
    """,
    unsafe_allow_html=True
)

st.dataframe(
    metrics.round(3),
    use_container_width=True,
    hide_index=True
)

# ---------------- VISUALIZATIONS ----------------
st.markdown("---")
st.markdown(
    '<div class="section-kicker">05 / VISUAL EXPLORATION</div>',
    unsafe_allow_html=True
)
st.header("Training and evaluation")

chart_left, chart_right = st.columns(2, gap="large")

with chart_left:
    path = OUTPUT_DIR / "model_rmse_comparison.png"
    if path.exists():
        st.image(
            str(path),
            caption="Model comparison by RMSE — lower is better",
            use_container_width=True
        )
    else:
        st.info("The RMSE chart appears after model training.")

with chart_right:
    path = OUTPUT_DIR / "ann_training_loss.png"
    if path.exists():
        st.image(
            str(path),
            caption="ANN training loss over iterations",
            use_container_width=True
        )
    else:
        st.info("The ANN loss chart appears after model training.")

path = OUTPUT_DIR / "actual_vs_predicted.png"
if path.exists():
    st.image(
        str(path),
        caption="Actual versus predicted scores for the best model",
        use_container_width=True
    )

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    STUDENT PERFORMANCE MINI PROJECT<br>
</div>
""", unsafe_allow_html=True)
