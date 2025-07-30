import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings

# Ignore specific warnings that might clutter the output
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=FutureWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=UserWarning, message='X does not have valid feature names')


st.set_page_config(layout="wide", page_title="Population Forecast App")

st.title("Population Forecasting & Environmental Analysis")
st.write("Upload your historical population data (CSV) to forecast future trends and identify optimal environmental conditions.")

# === File Uploader ===
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # === Basic Cleaning ===
        # Check if 'year' column exists before trying to convert
        if 'year' not in df.columns:
            st.error("Error: 'year' column not found in the uploaded CSV. Please ensure your CSV has a 'year' column.")
            st.stop()

        df['year'] = pd.to_datetime(df['year'], format='%Y')
        df.set_index('year', inplace=True)

        # Required columns check
        required_cols = ['pop', 'temperature', 'rainfall', 'habitat_index']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}. Please ensure your CSV includes all of: {', '.join(required_cols)}")
            st.stop()

        st.success("CSV loaded and basic cleaning applied!")
        st.subheader("Preview of your data:")
        st.dataframe(df.head())

        # === Split Data into Training and Testing Sets ===
        train_size = int(len(df) * 0.8)
        train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]

        # === Choose Forecasting Algorithm ===
        st.sidebar.header("Forecasting Settings")
        algorithm_choice = st.sidebar.selectbox(
            "Select Forecasting Algorithm:",
            ('ARIMA', 'SARIMA')
        ).upper() # Removed 'LSTM'

        future_steps = st.sidebar.slider("Number of years to forecast into the future:", 10, 200, 100)

        mse = None
        rmse = None
        forecast_label = ""
        forecast_test = None
        forecast_full = None
        extinction_year = None

        st.subheader(f"Running {algorithm_choice} Model...")

        with st.spinner(f"Training and forecasting with {algorithm_choice}... This might take a moment."):
            # Removed the entire 'if algorithm_choice == 'LSTM': ...' block
            if algorithm_choice == 'ARIMA':
                # === ARIMA Forecast ===
                arima_model = ARIMA(train_data['pop'], order=(3,1,1)).fit()
                arima_forecast_test = arima_model.forecast(steps=len(test_data))

                # === Evaluate ARIMA Model ===
                mse = mean_squared_error(test_data['pop'], arima_forecast_test)
                rmse = np.sqrt(mse)
                forecast_label = 'ARIMA Forecast'
                forecast_test = arima_forecast_test

                # === ARIMA Forecast (Full Data) for future plot ===
                arima_model_full = ARIMA(df['pop'], order=(3,1,1)).fit()
                forecast_full = arima_model_full.forecast(steps=future_steps)

                # === Extinction Year Detection (ARIMA) ===
                for i, val in enumerate(forecast_full):
                    if val <= 0:
                        extinction_year = df.index.year.max() + i + 1
                        break

            elif algorithm_choice == 'SARIMA':
                # === SARIMA Forecast ===
                sarima_model = SARIMAX(train_data['pop'], order=(1,1,1), seasonal_order=(1,1,0,12)).fit(disp=False)
                sarima_forecast_test = sarima_model.forecast(steps=len(test_data))

                # === Evaluate SARIMA Model ===
                mse = mean_squared_error(test_data['pop'], sarima_forecast_test)
                rmse = np.sqrt(mse)
                forecast_label = 'SARIMA Forecast'
                forecast_test = sarima_forecast_test

                # === SARIMA Forecast (Full Data) for future plot ===
                sarima_model_full = SARIMAX(df['pop'], order=(1,1,1), seasonal_order=(1,1,0,12)).fit(disp=False)
                forecast_full = sarima_model_full.forecast(steps=future_steps)

                # === Extinction Year Detection (SARIMA) ===
                for i, val in enumerate(forecast_full):
                    if val <= 0:
                        extinction_year = df.index.year.max() + i + 1
                        break
            else:
                st.error("Invalid algorithm choice. Please select ARIMA or SARIMA.") # Updated error message
                st.stop()

        # === Plot Forecast ===
        years_future = np.arange(df.index.year.max() + 1, df.index.year.max() + 1 + future_steps)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index.year, df['pop'], label='Historical Population', linewidth=2)
        ax.plot(years_future, forecast_full, label=forecast_label, linestyle='--', color='orange')
        ax.set_xlabel('Year')
        ax.set_ylabel('Population')
        ax.set_title(f'Population Forecast using {algorithm_choice}')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # === Display Evaluation Metrics ===
        st.subheader("📈 Model Evaluation Metrics:")
        if mse is not None and rmse is not None:
            st.info(f"{algorithm_choice} - Mean Squared Error (MSE): {mse:.2f}, Root Mean Squared Error (RMSE): {rmse:.2f}")
        else:
            st.warning("Evaluation metrics could not be calculated.")

        # === Show Predicted Extinction Year ===
        st.subheader("🗓️ Extinction Prediction:")
        if extinction_year:
            st.warning(f"⚠️ Based on the {algorithm_choice} model, extinction is predicted in the year: **{extinction_year}**")
        else:
            st.success(f"✅ The {algorithm_choice} model did not predict extinction within the next {future_steps} years.")

        st.markdown("---")
        st.subheader("🌿 Environmental Conditions for Thriving Population")

        # === Thriving Population Range Input ===
        col1, col2 = st.columns(2)
        with col1:
            low_thriving = st.number_input("Enter lower bound of thriving population range:", min_value=0, value=int(df['pop'].min()), step=1)
        with col2:
            high_thriving = st.number_input("Enter upper bound of thriving population range:", min_value=0, value=int(df['pop'].max()), step=1)

        if low_thriving >= high_thriving:
            st.error("Lower bound must be less than upper bound.")
        else:
            # === Filter and Report Environmental Vectors ===
            thriving_years = df[(df['pop'] >= low_thriving) & (df['pop'] <= high_thriving)]

            if thriving_years.empty:
                st.info(f"No years found where population was between {low_thriving} and {high_thriving}.")
            else:
                environmental_means = thriving_years[['temperature', 'rainfall', 'habitat_index']].mean().round(2)
                st.subheader("🧠 Average Environmental and Climate Conditions for Thriving Population:")
                st.write(f"**Temperature:** {environmental_means['temperature']} °C")
                st.write(f"**Rainfall:** {environmental_means['rainfall']} mm")
                st.write(f"**Habitat Index:** {environmental_means['habitat_index']}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.write("Please ensure your CSV file is correctly formatted with 'year', 'pop', 'temperature', 'rainfall', and 'habitat_index' columns.")

else:
    st.info("Please upload a CSV file to begin.")
