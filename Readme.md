# Retail Demand Forecasting Engine

A comprehensive time-series forecasting system that predicts weekly or monthly product demand for different SKUs, factoring in past sales, promotions, seasonality, and regional trends.

## Features

- **Advanced Time-Series Forecasting**: Uses Random Forest with engineered features
- **Multi-SKU Support**: Forecast demand for different products simultaneously
- **Regional Analysis**: Account for regional trends and patterns
- **Seasonality Detection**: Captures seasonal patterns and cyclical trends
- **Promotion Impact**: Factor in promotional activities and marketing spend
- **Interactive Dashboard**: User-friendly Streamlit interface
- **Confidence Intervals**: Provides uncertainty quantification for forecasts
- **Export Functionality**: Download forecasts as CSV files

##  Requirements

- Python 3.8+
- See `requirements.txt` for complete dependencies

##  Installation

1. **Clone or download the project files**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Kaggle API (if using Kaggle dataset):**
   - Create a Kaggle account and generate API key
   - Place `kaggle.json` in `~/.kaggle/` directory
   - Or set up the dataset path manually in `train.py`

##  Usage

### Step 1: Train the Model

Run the training script to build the forecasting model:

```bash
python train.py
```

This will:
- Load the retail dataset from Kaggle
- Perform feature engineering
- Train a Random Forest model
- Save the model to `demand_forecasting_model.joblib`
- Generate sample data for the dashboard

### Step 2: Launch the Dashboard

Start the Streamlit application:

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## How It Works

### 1. Data Processing
- **Time-Series Features**: Date components, cyclical encodings
- **Lag Features**: Historical sales at different time intervals
- **Rolling Statistics**: Moving averages and standard deviations
- **Categorical Encoding**: Product and region encodings

### 2. Model Training
- **Algorithm**: Random Forest Regressor
- **Features**: 20+ engineered features
- **Validation**: Time-series split for realistic evaluation
- **Metrics**: MAE, RMSE, and MAPE

### 3. Forecasting
- **Multi-step Ahead**: Predict multiple future periods
- **Confidence Intervals**: Quantify prediction uncertainty
- **Business Parameters**: Include promotions, marketing spend, pricing

## Dashboard Features

### Main Interface
- **Product Selection**: Choose from available SKUs
- **Region Selection**: Select geographic region
- **Date Range**: Set forecast start date and duration
- **Business Parameters**: Adjust promotions, marketing spend, and pricing

### Visualizations
- **Forecast Plot**: Main demand prediction with confidence intervals
- **Historical Analysis**: Recent sales trends and distributions
- **Feature Importance**: Model interpretability insights

### Export Options
- **CSV Download**: