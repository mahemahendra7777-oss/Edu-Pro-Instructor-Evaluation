import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Edu-Pro Instructor Evaluation Dashboard",
    page_icon="🎓",
    layout="wide",
)


def generate_sample_data(n_rows: int = 800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    instructors = [f"Instructor {i:02d}" for i in range(1, 13)]
    courses = [
        "Python & Data Science",
        "Web Development",
        "Cloud & DevOps",
        "UI/UX Fundamentals",
        "Database Systems",
    ]
    exp_levels = ["0-2 yrs", "3-5 yrs", "6-10 yrs", "10+ yrs"]

    rows = []
    for idx in range(1, n_rows + 1):
        inst = rng.choice(instructors)
        course = rng.choice(courses)
        exp = rng.choice(exp_levels, p=[0.2, 0.35, 0.3, 0.15])
        base_rating = {"0-2 yrs": 3.8, "3-5 yrs": 4.1, "6-10 yrs": 4.4, "10+ yrs": 4.6}[exp]
        rating = float(np.clip(rng.normal(base_rating, 0.45), 1.0, 5.0))
        engagement = float(np.clip(rating * 20 + rng.normal(0, 6), 20, 100))
        completion = float(np.clip(rating * 18 + rng.normal(5, 8), 25, 100))
        students = int(rng.integers(25, 250))

        rows.append({
            "evaluation_id": idx,
            "instructor_name": inst,
            "course_name": course,
            "experience_years": exp,
            "rating": round(rating, 2),
            "engagement_score": round(engagement, 1),
            "completion_rate": round(completion, 1),
            "enrolled_students": students,
        })
    return pd.DataFrame(rows)


@st.cache_data
def load_data():
    candidates = [
        "EduPro Online Platform.xlsx",
        "data/EduPro Online Platform.xlsx",
    ]
    found_path = next((p for p in candidates if os.path.exists(p)), None)

    if found_path:
        df = pd.read_excel(found_path)
        is_sample = False
    else:
        df = generate_sample_data()
        is_sample = True

    df.columns = [str(c).strip() for c in df.columns]
    return df, is_sample, (found_path or "Sample Fallback Dataset")


df, is_sample, source_label = load_data()

st.title("🎓 Edu-Pro Online Platform — Instructor Evaluation")
st.caption(f"Source: `{source_label}` | Rows: `{len(df):,}` | Columns: `{len(df.columns)}`")

if is_sample:
    st.info(
        "Showing sample data because `EduPro Online Platform.xlsx` was not detected. "
        "Keep `EduPro Online Platform.xlsx` in the repo root to analyze your full dataset."
    )

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [
    c for c in df.select_dtypes(exclude=[np.number]).columns.tolist()
    if 1 < df[c].nunique(dropna=True) <= 50
]

if not numeric_cols:
    st.error("No numeric columns found in the dataset.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters & Metric Controls")
selected_metric = st.sidebar.selectbox("Primary Numeric Metric", options=numeric_cols, index=0)

filtered_df = df.copy()
for col in categorical_cols[:3]:
    options = sorted(filtered_df[col].dropna().astype(str).unique().tolist())
    chosen = st.sidebar.multiselect(f"Filter by {col}", options=options, default=options)
    if chosen:
        filtered_df = filtered_df[filtered_df[col].astype(str).isin(chosen)]

if filtered_df.empty:
    st.warning("No rows match the selected filters.")
    st.stop()

# KPI row
k1, k2, k3, k4 = st.columns(4)
k1.metric("Records Shown", f"{len(filtered_df):,}")
k2.metric(f"Average {selected_metric}", f"{filtered_df[selected_metric].mean():.2f}")
k3.metric(f"Median {selected_metric}", f"{filtered_df[selected_metric].median():.2f}")
k4.metric(f"Max {selected_metric}", f"{filtered_df[selected_metric].max():.2f}")

st.divider()

# Chart 1: Breakdown by category
if categorical_cols:
    group_col = st.selectbox("Group By Category", options=categorical_cols, index=0)
    agg_df = (
        filtered_df.groupby(group_col, as_index=False)[selected_metric]
        .mean()
        .sort_values(selected_metric, ascending=False)
    )
    fig_bar = px.bar(
        agg_df,
        x=group_col,
        y=selected_metric,
        title=f"Average {selected_metric} by {group_col}",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# Chart 2: Distribution
fig_hist = px.histogram(
    filtered_df,
    x=selected_metric,
    nbins=25,
    title=f"Distribution of {selected_metric}",
)
st.plotly_chart(fig_hist, use_container_width=True)

# Chart 3: Relationship between two numeric columns (if available)
if len(numeric_cols) >= 2:
    c_x, c_y = st.columns(2)
    x_num = c_x.selectbox("X-Axis Numeric", options=numeric_cols, index=0)
    y_num = c_y.selectbox(
        "Y-Axis Numeric", options=numeric_cols, index=min(1, len(numeric_cols) - 1)
    )
    color_col = categorical_cols[0] if categorical_cols else None
    fig_scatter = px.scatter(
        filtered_df,
        x=x_num,
        y=y_num,
        color=color_col,
        title=f"{y_num} vs. {x_num}",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# Raw data preview
with st.expander("Preview Filtered Data"):
    st.dataframe(filtered_df.head(100), use_container_width=True)
