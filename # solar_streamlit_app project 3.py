# solar_streamlit_app.py

# solar_streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

st.set_page_config(page_title="Solar Power Generation", layout="wide")

st.title("🌞 Solar Power Generation Prediction (Random Forest)")

# ------------------------
# File Upload
# ------------------------
uploaded_file = st.file_uploader(r"https://raw.githubusercontent.com/Mounikakolli7/project-3/main/solarpowergeneration%20(1).csv", type=["csv"])
if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # Remove extra spaces

    # Show dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Show all column names
    st.write("Column names:", df.columns.tolist())

    # Auto-detect target column
    target_candidates = [col for col in df.columns if "power" in col.lower()]
    if not target_candidates:
        st.error("No column found containing 'power'. Please check your dataset.")
        st.stop()
    target_col = target_candidates[0]
    st.write(f"Using target column: **{target_col}**")

    # ------------------------
    # EDA Section
    # ------------------------
    st.subheader("Dataset Summary")
    st.write(f"Shape: {df.shape}")
    st.write(df.describe())

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    # ------------------------
    # Handle Missing Values
    # ------------------------
    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # ------------------------
    # Model Training (Random Forest)
    # ------------------------
    st.subheader("Model Training")

    X = df.drop(target_col, axis=1)  # Features
    y = df[target_col]               # Target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Model performance
    st.write("**Model Performance:**")
    st.write(f"R² Score: {r2_score(y_test, y_pred):.3f}")
    st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
    st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")

    # Save model
    joblib.dump(model, "solar_model.pkl")

    # ------------------------
    # Prediction Section
    # ------------------------
    st.subheader("🔮 Predict Power Generation")

    distance_to_solar_noon = st.number_input("Distance to Solar Noon")

