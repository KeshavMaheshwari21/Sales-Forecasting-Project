import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import plotly.graph_objects as go
from datetime import timedelta

st.title("📊 Sales Forecasting with XGBoost")

file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if file:
    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    st.write("Data Preview:")
    st.dataframe(df)

    date_col = st.selectbox("Select the date column", df.columns)
    target_col = st.selectbox("Select the column to forecast", df.columns)

    n_future = st.number_input("Number of future days to forecast", min_value=1, max_value=365, value=30)

    if st.button("Run Forecast"):
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
        df = df.sort_values(date_col)

        # Feature engineering
        df["day"] = df[date_col].dt.day
        df["month"] = df[date_col].dt.month
        df["year"] = df[date_col].dt.year
        df["dayofweek"] = df[date_col].dt.dayofweek

        # Lag features
        for i in range(1, 8):
            df[f"lag_{i}"] = df[target_col].shift(i)

        df = df.dropna()

        # X and y
        features = ["day", "month", "year", "dayofweek"] + [f"lag_{i}" for i in range(1, 8)]
        X = df[features]
        y = df[target_col]

        model = XGBRegressor()
        model.fit(X, y)

        last_known = df.iloc[-1]
        future_dates = [last_known[date_col] + timedelta(days=i) for i in range(1, n_future + 1)]
        future_data = []

        prev_values = list(df[target_col].values[-7:])

        for date in future_dates:
            row = {
                "day": date.day,
                "month": date.month,
                "year": date.year,
                "dayofweek": date.dayofweek,
            }
            for i in range(1, 8):
                row[f"lag_{i}"] = prev_values[-i]
            X_pred = pd.DataFrame([row])
            y_pred = model.predict(X_pred)[0]
            prev_values.append(y_pred)
            future_data.append((date, y_pred))

        # Plotly Graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df[date_col], y=df[target_col], name='Historical'))
        fig.add_trace(go.Scatter(x=[d[0] for d in future_data], y=[d[1] for d in future_data], name='Forecast'))
        fig.update_layout(title="XGBoost Sales Forecast", xaxis_title="Date", yaxis_title="Value", template="plotly_white")

        st.plotly_chart(fig, use_container_width=True)
