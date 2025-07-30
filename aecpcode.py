import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings

# --- Configuration and Initial Setup ---
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=FutureWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=UserWarning, message='X does not have valid feature names')

st.set_page_config(layout="wide", page_title="Population Forecast App", icon="📈") # Added an icon

# --- Helper Functions ---

@st.cache_data
def load_data(uploaded_file):
    """
    Loads and performs initial cleaning on the uploaded CSV data.
    Uses st.cache_data to avoid reloading on every rerun.
    """
    df = pd.read_csv(uploaded_file)
    
    # Standardize column names (optional but good practice)
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]

    # Validate essential columns
    required_cols = ['year', 'pop', 'temperature', 'rainfall', 'habitat_index']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}. Please ensure your CSV includes all of: {', '.join(required_cols)}")
        st.stop() # Stop execution if critical columns are missing

    try:
        df['year'] = pd.to_datetime(df['year'], format='%Y')
        df.set_index('year', inplace=True)
    except Exception as e:
        st.error(f"Error converting 'year' column to datetime. Please ensure it's in a 'YYYY' format. Details: {e}")
        st.stop()

    # Basic data validation: Ensure 'pop' is numeric and non-negative
    if not pd.api.types.is_numeric_dtype(df['pop']):
        st.error("Error: 'pop' column must contain numeric values.")
        st.stop()
    if (df['pop'] < 0).any():
        st.warning("Warning: Population values contain negative numbers. These will be treated as 0 for forecasting purposes, but consider checking your data.")
        df['pop'] = df['pop'].clip(lower=0) # Clip negative populations to 0


    return df

@st.cache_resource
def train_and_forecast_arima(train_data, full_data, future_steps, order=(3,1,1)):
    """Trains and forecasts using ARIMA model."""
    try:
        arima_model = ARIMA(train_data['pop'], order=order).fit()
        forecast_test = arima_model.forecast(steps=len(test_data))

        arima_model_full = ARIMA(full_data['pop'], order=order).fit()
        forecast_full = arima_model_full.forecast(steps=future_steps)
        return forecast_test, forecast_full
    except Exception as e:
        st.error(f"Error training/forecasting with ARIMA: {e}. Try adjusting model parameters.")
        return None, None

@st.cache_resource
def train_and_forecast_sarima(train_data, full_data, future_steps, order=(1,1,1), seasonal_order=(1,1,0,12)):
    """Trains and forecasts using SARIMA model."""
    try:
        sarima_model = SARIMAX(train_data['pop'], order=order, seasonal_order=seasonal_order).fit(disp=False)
        forecast_test = sarima_model.forecast(steps=len(test_data))

        sarima_model_full = SARIMAX(full_data['pop'], order=order, seasonal_order=seasonal_order).fit(disp=False)
        forecast_full = sarima_model_full.forecast(steps=future_steps)
        return forecast_test, forecast_full
    except Exception as e:
        st.error(f"Error training/forecasting with SARIMA: {e}. Try adjusting model parameters or seasonal order.")
        return None, None

def calculate_extinction_year(start_year, forecast_values):
    """Calculates the predicted extinction year based on forecast values."""
    for i, val in enumerate(forecast_values):
        if val <= 0:
            return start_year + i + 1
    return None

# --- Main Streamlit App Logic ---

st.title("Population Forecasting & Environmental Analysis")
st.write("Upload your historical population data (CSV) to forecast future trends and identify optimal environmental conditions.")

# === File Uploader ===
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = load_data(uploaded_file) # Use cached function

    if df is not None: # Check if data loading and validation were successful
        st.success("CSV loaded and basic cleaning applied!")
        st.subheader("Preview of your data:")
        st.dataframe(df.head())

        # === Split Data into Training and Testing Sets ===
        train_size = int(len(df) * 0.8)
        
        if train_size == 0 or train_size == len(df):
            st.error("Dataset is too small to split into training and testing sets. Please upload a dataset with more rows.")
            st.stop()

        train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]

        # === Choose Forecasting Algorithm ===
        st.sidebar.header("Forecasting Settings")
        algorithm_choice = st.sidebar.selectbox(
            "Select Forecasting Algorithm:",
            ('ARIMA', 'SARIMA')
        ).upper()

        future_steps = st.sidebar.slider("Number of years to forecast into the future:", 10, 200, 50) # Default to 50 for faster initial run

        mse = None
        rmse = None
        forecast_label = ""
        forecast_test = None
        forecast_full = None
        extinction_year = None

        st.subheader(f"Running {algorithm_choice} Model...")

        with st.spinner(f"Training and forecasting with {algorithm_choice}... This might take a moment."):
            if algorithm_choice == 'ARIMA':
                forecast_test, forecast_full = train_and_forecast_arima(train_data, df, future_steps)
                forecast_label = 'ARIMA Forecast'
            elif algorithm_choice == 'SARIMA':
                forecast_test, forecast_full = train_and_forecast_sarima(train_data, df, future_steps)
                forecast_label = 'SARIMA Forecast'
            else:
                st.error("Invalid algorithm choice. Please select ARIMA or SARIMA.")
                st.stop()

            if forecast_test is not None and forecast_full is not None:
                # Calculate evaluation metrics only if forecast_test is not empty and matches test_data length
                if len(test_data['pop']) > 0 and len(forecast_test) == len(test_data['pop']):
                    mse = mean_squared_error(test_data['pop'], forecast_test)
                    rmse = np.sqrt(mse)
                else:
                    st.warning("Could not calculate test set metrics: Test data or forecast length mismatch. Showing full data forecast only.")

                extinction_year = calculate_extinction_year(df.index.year.max(), forecast_full)
            else:
                st.warning("Forecasting failed. Please check your data and model parameters.")

        # === Plot Forecast ===
        if forecast_full is not None:
            years_future = pd.to_datetime(np.arange(df.index.year.max() + 1, df.index.year.max() + 1 + future_steps), format='%Y')

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df.index, df['pop'], label='Historical Population', linewidth=2)
            ax.plot(years_future, forecast_full, label=forecast_label, linestyle='--', color='orange')
            
            # Add a horizontal line at 0 for extinction visualization
            ax.axhline(y=0, color='r', linestyle=':', label='Extinction Line')

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
                st.info("Evaluation metrics not available for this run (e.g., due to insufficient test data or forecast failure).")

            # === Show Predicted Extinction Year ===
            st.subheader("🗓️ Extinction Prediction:")
            if extinction_year:
                st.warning(f"⚠️ Based on the {algorithm_choice} model, extinction is predicted in the year: **{extinction_year}**")
            else:
                st.success(f"✅ The {algorithm_choice} model did not predict extinction within the next {future_steps} years.")
        else:
            st.error("Cannot plot forecast as model training or prediction failed.")

        st.markdown("---")
        st.subheader("🌿 Environmental Conditions for Thriving Population")

        # === Thriving Population Range Input ===
        col1, col2 = st.columns(2)
        with col1:
            # Set dynamic default values based on loaded data
            low_thriving = st.number_input(
                "Enter lower bound of thriving population range:", 
                min_value=0, 
                value=int(df['pop'].min() * 0.9) if df['pop'].min() > 0 else 0, # Slightly below min
                step=1
            )
        with col2:
            high_thriving = st.number_input(
                "Enter upper bound of thriving population range:", 
                min_value=0, 
                value=int(df['pop'].max() * 1.1), # Slightly above max
                step=1
            )

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

else:
    st.info("Please upload a CSV file to begin.")
