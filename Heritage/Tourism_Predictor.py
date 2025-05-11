import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go

# Load data
df = pd.read_csv("data/foot_fall.csv")

# Prepare datetime column
df["ds"] = pd.to_datetime(df["Year"].astype(str), format="%Y")
df.rename(columns={"Visitors": "y"}, inplace=True)

# UI title
st.title("📈 Cultural Tourism Seasonality Predictor")

# Site selection
sites = sorted(df["Place"].dropna().unique())
selected_site = st.selectbox("Choose a cultural site", sites)

# Filter and clean data
site_df = df[df["Place"] == selected_site][["ds", "y"]].dropna().sort_values("ds")

# Show warning if too few data points
if len(site_df) < 2:
    st.warning(f"🚫 Not enough data for '{selected_site}' to generate a forecast (only {len(site_df)} valid data point{'s' if len(site_df) != 1 else ''}). Try another site.")
    st.stop()

# Train Prophet
model = Prophet(yearly_seasonality=True)
model.fit(site_df)

# Forecast future values (12 years forward)
future = model.make_future_dataframe(periods=12, freq='Y')
forecast = model.predict(future)

# Forecast Plot
st.subheader(f"📊 Forecasted Visitors for {selected_site}")
fig = plot_plotly(model, forecast)
st.plotly_chart(fig)

# Historical + Forecast Line Chart
st.subheader("📉 Visitor Trend")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=site_df["ds"], y=site_df["y"], name="Historical", mode='lines+markers'))
fig2.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast", mode='lines'))
st.plotly_chart(fig2)
