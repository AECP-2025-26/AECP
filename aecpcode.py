import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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
            ('ARIMA', 'SARIMA', 'LSTM')
        ).upper()

        future_steps = st.sidebar.slider("Number of years to forecast into the future:", 10, 200, 100)

        mse = None
        rmse = None
        forecast_label = ""
        forecast_test = None
        forecast_full = None
        extinction_year = None

        st.subheader(f"Running {algorithm_choice} Model...")

        with st.spinner(f"Training and forecasting with {algorithm_choice}... This might take a moment."):
            if algorithm_choice == 'LSTM':
                # === Normalize population for LSTM ===
                scaler = MinMaxScaler()
                train_data_scaled = scaler.fit_transform(train_data[['pop']])
                test_data_scaled = scaler.transform(test_data[['pop']])

                window_size = 5

                def create_sequences(data, window):
                    X, y = [], []
                    for i in range(len(data) - window):
                        X.append(data[i:i+window])
                        y.append(data[i+window])
                    return np.array(X), np.array(y)

                X_train, y_train = create_sequences(train_data_scaled, window_size)
                X_test, y_test = create_sequences(test_data_scaled, window_size)

                # Ensure X_train and X_test have correct dimensions for LSTM
                if X_train.shape[0] == 0 or X_test.shape[0] == 0:
                    st.warning(f"Not enough data to create sequences for LSTM with window size {window_size}. Please check your data length.")
                    st.stop()

                # === Build and Train LSTM Model ===
                model = Sequential([
                    LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)

                # === Evaluate LSTM Model ===
                lstm_predictions_scaled = model.predict(X_test, verbose=0)
                lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
                mse = mean_squared_error(test_data['pop'].iloc[window_size:], lstm_predictions)
                rmse = np.sqrt(mse)
                forecast_label = 'LSTM Forecast'
                forecast_test = lstm_predictions

                # === Forecast with LSTM (Full Data) for future plot ===
                data_scaled_full = scaler.fit_transform(df[['pop']]) # Re-fit scaler on full data
                X_full, y_full = create_sequences(data_scaled_full, window_size)

                if X_full.shape[0] == 0:
                    st.warning(f"Not enough data to create sequences for full LSTM forecast with window size {window_size}. Please check your data length.")
                    st.stop()

                model_full = Sequential([
                    LSTM(64, activation='relu', input_shape=(X_full.shape[1], X_full.shape[2])),
                    Dense(1)
                ])
                model_full.compile(optimizer='adam', loss='mse')
                model_full.fit(X_full, y_full, epochs=50, batch_size=8, verbose=0)

                last_seq_full = data_scaled_full[-window_size:]
                lstm_preds_full = []
                for _ in range(future_steps):
                    input_seq_full = last_seq_full.reshape(1, window_size, 1)
                    pred_full = model_full.predict(input_seq_full, verbose=0)
                    lstm_preds_full.append(pred_full[0, 0])
                    last_seq_full = np.append(last_seq_full[1:], pred_full.reshape(1, 1), axis=0)
                forecast_full = scaler.inverse_transform(np.array(lstm_preds_full).reshape(-1, 1))

                # === Extinction Year Detection (LSTM) ===
                for i, val in enumerate(forecast_full):
                    if val[0] <= 0:
                        extinction_year = df.index.year.max() + i + 1
                        break

            elif algorithm_choice == 'ARIMA':
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
                st.error("Invalid algorithm choice. Please select ARIMA, SARIMA, or LSTM.")
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

