import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Demand Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stSelectbox > div > div > select {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None

@st.cache_data
def load_models():
    """Load pre-trained models"""
    try:
        # Load Prophet model with pickle
        with open('best_prophet_model.pkl', 'rb') as f:
            prophet_model = pickle.load(f)
        
        # Load Random Forest and XGBoost models with joblib
        rf_model = joblib.load('best_random_forest_model.pkl')
        xgb_model = joblib.load('best_xgboost_model.pkl')
        
        return prophet_model, rf_model, xgb_model
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

@st.cache_data
def load_retail_data():
    """Load the actual retail store inventory dataset"""
    try:
        # Load the CSV file
        df = pd.read_csv('retail_store_inventory.csv')
        
        # Convert Date column to datetime and set as index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Clean column names to match your feature engineering
        column_mapping = {
            'Units Sold': 'Units_Sold',
            'Inventory Level': 'Inventory_Level',
            'Product ID': 'Product_ID',
            'Store ID': 'Store_ID',
            'Holiday/Promotion': 'Holiday_Promotion',
            'Competitor Pricing': 'Competitor_Pricing',
            'Weather Condition': 'Weather_Condition'
        }
        
        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        # Handle missing values
        df.fillna(method='ffill', inplace=True)
        
        return df
        
    except FileNotFoundError:
        st.error("❌ retail_store_inventory.csv not found. Please upload your dataset.")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def prepare_features_for_ml(df, product_id, store_id):
    """Prepare features for ML models (RF and XGBoost) based on your actual feature engineering"""
    # Filter data for specific product and store
    filtered_df = df[(df['Product_ID'] == product_id) & (df['Store_ID'] == store_id)].copy()
    
    if len(filtered_df) == 0:
        return None
    
    # Create time-based features
    filtered_df['Month'] = filtered_df.index.month
    filtered_df['Day'] = filtered_df.index.day
    filtered_df['Week'] = filtered_df.index.isocalendar().week
    filtered_df['Year'] = filtered_df.index.year
    
    # Create seasonality features
    filtered_df['Seasonality_Spring'] = (filtered_df['Month'].isin([3, 4, 5])).astype(int)
    filtered_df['Seasonality_Summer'] = (filtered_df['Month'].isin([6, 7, 8])).astype(int)
    filtered_df['Seasonality_Winter'] = (filtered_df['Month'].isin([12, 1, 2])).astype(int)
    
    # Create differenced features
    filtered_df['Units_Sold_Differenced'] = filtered_df['Units_Sold'].diff()
    
    # Create lag features
    for lag in [1, 7, 30]:
        filtered_df[f'Lag_{lag}'] = filtered_df['Units_Sold'].shift(lag)
    
    # Create rolling window features
    for window in [7, 30]:
        filtered_df[f'Rolling_Mean_{window}'] = filtered_df['Units_Sold'].rolling(window=window).mean()
        filtered_df[f'Rolling_Std_{window}'] = filtered_df['Units_Sold'].rolling(window=window).std()
    
    # One-hot encode categorical variables
    categorical_cols = ['Category', 'Region', 'Weather_Condition']
    for col in categorical_cols:
        if col in filtered_df.columns:
            dummies = pd.get_dummies(filtered_df[col], prefix=col)
            filtered_df = pd.concat([filtered_df, dummies], axis=1)
    
    # Handle Product_ID and Store_ID one-hot encoding
    if 'Product_ID' in filtered_df.columns:
        product_dummies = pd.get_dummies(filtered_df['Product_ID'], prefix='Product ID')
        filtered_df = pd.concat([filtered_df, product_dummies], axis=1)
    
    if 'Store_ID' in filtered_df.columns:
        store_dummies = pd.get_dummies(filtered_df['Store_ID'], prefix='Store ID')
        filtered_df = pd.concat([filtered_df, store_dummies], axis=1)
    
    # Convert boolean columns to int
    bool_cols = ['Holiday_Promotion', 'Competitor_Pricing']
    for col in bool_cols:
        if col in filtered_df.columns:
            if filtered_df[col].dtype == 'bool':
                filtered_df[col] = filtered_df[col].astype(int)
            elif filtered_df[col].dtype == 'object':
                filtered_df[col] = filtered_df[col].astype(bool).astype(int)
    
    # Drop original categorical columns
    cols_to_drop = categorical_cols + ['Product_ID', 'Store_ID']
    filtered_df.drop(cols_to_drop, axis=1, inplace=True, errors='ignore')
    
    # Drop rows with NaN values
    filtered_df.dropna(inplace=True)
    
    return filtered_df

def prepare_prophet_data(df, product_id, store_id):
    """Prepare data for Prophet model"""
    # Filter data for specific product and store
    filtered_df = df[(df['Product_ID'] == product_id) & (df['Store_ID'] == store_id)].copy()
    
    if len(filtered_df) == 0:
        return None
    
    # Prophet requires 'ds' and 'y' columns
    prophet_df = filtered_df[['Units_Sold']].copy()
    prophet_df.reset_index(inplace=True)
    prophet_df.columns = ['ds', 'y']
    
    return prophet_df

def make_ml_predictions(model, data, model_type, forecast_days=60):
    """Make predictions using ML models (RF/XGBoost) matching your feature engineering approach"""
    if data is None or len(data) == 0:
        return None, None
    
    # Get the last row of data for prediction
    last_row = data.iloc[-1:].copy()
    
    # Expected feature columns based on your model training
    expected_features = [
        'Inventory_Level', 'Units_Ordered', 'Demand_Forecast', 'Price', 'Discount',
        'Holiday_Promotion', 'Competitor_Pricing', 'Store ID_S002', 'Store ID_S003',
        'Store ID_S004', 'Store ID_S005', 'Product ID_P0002', 'Product ID_P0003',
        'Product ID_P0004', 'Product ID_P0005', 'Product ID_P0006', 'Product ID_P0007',
        'Product ID_P0008', 'Product ID_P0009', 'Product ID_P0010', 'Product ID_P0011',
        'Product ID_P0012', 'Product ID_P0013', 'Product ID_P0014', 'Product ID_P0015',
        'Product ID_P0016', 'Product ID_P0017', 'Product ID_P0018', 'Product ID_P0019',
        'Product ID_P0020', 'Category_Electronics', 'Category_Furniture',
        'Category_Groceries', 'Category_Toys', 'Region_North', 'Region_South',
        'Region_West', 'Weather_Condition_Rainy', 'Weather_Condition_Snowy',
        'Weather_Condition_Sunny', 'Seasonality_Spring', 'Seasonality_Summer',
        'Seasonality_Winter', 'Units_Sold_Differenced', 'Month', 'Day', 'Week',
        'Year', 'Lag_1', 'Lag_7', 'Lag_30', 'Rolling_Mean_7', 'Rolling_Std_7',
        'Rolling_Mean_30', 'Rolling_Std_30'
    ]
    
    # Prepare features - use only available columns that match expected features
    available_features = [col for col in expected_features if col in data.columns]
    
    if len(available_features) == 0:
        st.error("No matching features found for prediction")
        return None, None
    
    # Use available features for prediction
    X_last = last_row[available_features].fillna(0)
    
    # Generate future predictions
    future_predictions = []
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), 
                                periods=forecast_days, freq='D')
    
    current_row = X_last.copy()
    
    for i in range(forecast_days):
        try:
            # Make prediction
            pred = model.predict(current_row)[0]
            future_predictions.append(max(0, pred))  # Ensure non-negative predictions
            
            # Update lag features for next prediction
            if 'Lag_1' in current_row.columns:
                if 'Lag_7' in current_row.columns:
                    current_row['Lag_7'] = current_row['Lag_1'].iloc[0] if i >= 6 else current_row['Lag_7'].iloc[0]
                if 'Lag_30' in current_row.columns:
                    current_row['Lag_30'] = current_row['Lag_1'].iloc[0] if i >= 29 else current_row['Lag_30'].iloc[0]
                current_row['Lag_1'] = pred
            
            # Update rolling features (simplified)
            if 'Rolling_Mean_7' in current_row.columns:
                current_row['Rolling_Mean_7'] = pred
            if 'Rolling_Mean_30' in current_row.columns:
                current_row['Rolling_Mean_30'] = pred
            
            # Update time-based features
            current_date = future_dates[i]
            if 'Month' in current_row.columns:
                current_row['Month'] = current_date.month
            if 'Day' in current_row.columns:
                current_row['Day'] = current_date.day
            if 'Week' in current_row.columns:
                current_row['Week'] = current_date.isocalendar().week
            if 'Year' in current_row.columns:
                current_row['Year'] = current_date.year
            
            # Update seasonality features
            if 'Seasonality_Spring' in current_row.columns:
                current_row['Seasonality_Spring'] = 1 if current_date.month in [3, 4, 5] else 0
            if 'Seasonality_Summer' in current_row.columns:
                current_row['Seasonality_Summer'] = 1 if current_date.month in [6, 7, 8] else 0
            if 'Seasonality_Winter' in current_row.columns:
                current_row['Seasonality_Winter'] = 1 if current_date.month in [12, 1, 2] else 0
                
        except Exception as e:
            st.error(f"Error in prediction step {i}: {e}")
            future_predictions.append(0)
    
    return future_dates, future_predictions

def make_prophet_predictions(model, data, forecast_days=60):
    """Make predictions using Prophet model"""
    if data is None or len(data) == 0:
        return None, None
    
    try:
        # Create future dataframe
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        
        # Get only the forecasted period
        forecast_period = forecast.tail(forecast_days)
        future_dates = pd.to_datetime(forecast_period['ds'])
        future_predictions = forecast_period['yhat'].values
        
        # Get confidence intervals
        lower_bound = forecast_period['yhat_lower'].values
        upper_bound = forecast_period['yhat_upper'].values
        
        return future_dates, future_predictions, lower_bound, upper_bound
        
    except Exception as e:
        st.error(f"Error in Prophet prediction: {e}")
        return None, None, None, None

def create_forecast_chart(historical_data, future_dates, future_predictions, 
                         lower_bound=None, upper_bound=None, title="Demand Forecast"):
    """Create interactive forecast chart"""
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data['Units_Sold'],
        mode='lines',
        name='Historical Demand',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_predictions,
        mode='lines',
        name='Forecasted Demand',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    # Add confidence intervals if available
    if lower_bound is not None and upper_bound is not None:
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=lower_bound,
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(255,127,14,0.3)',
            fill='tonexty',
            name='Confidence Interval',
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Units Sold',
        hovermode='x unified',
        showlegend=True,
        height=400
    )
    
    return fig

def create_inventory_gap_chart(future_dates, future_predictions, inventory_levels):
    """Create inventory gap analysis chart"""
    fig = go.Figure()
    
    # Add forecasted demand
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_predictions,
        mode='lines',
        name='Forecasted Demand',
        line=dict(color='#ff7f0e', width=2)
    ))
    
    # Add inventory levels
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=inventory_levels,
        mode='lines',
        name='Inventory Levels',
        line=dict(color='#2ca02c', width=2)
    ))
    
    # Add overstock area
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=np.maximum(inventory_levels, future_predictions),
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_predictions,
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(44,160,44,0.3)',
        fill='tonexty',
        name='Overstock',
        hoverinfo='skip'
    ))
    
    # Add stockout risk area
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=np.minimum(inventory_levels, future_predictions),
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(214,39,40,0.3)',
        fill='tonexty',
        name='Stockout Risk',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title='Inventory Gap Analysis',
        xaxis_title='Date',
        yaxis_title='Units',
        hovermode='x unified',
        showlegend=True,
        height=400
    )
    
    return fig

def create_model_comparison_chart(future_dates, predictions_dict):
    """Create model comparison chart"""
    fig = go.Figure()
    
    colors = ['#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (model_name, predictions) in enumerate(predictions_dict.items()):
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines',
            name=f'{model_name} Forecast',
            line=dict(color=colors[i], width=2)
        ))
    
    fig.update_layout(
        title='Model Comparison',
        xaxis_title='Date',
        yaxis_title='Units Sold',
        hovermode='x unified',
        showlegend=True,
        height=400
    )
    
    return fig

# Main app
def main():
    st.markdown('<h1 class="main-header">Demand Forecasting Dashboard</h1>', unsafe_allow_html=True)
    
    # Load models
    prophet_model, rf_model, xgb_model = load_models()
    
    if prophet_model is None or rf_model is None or xgb_model is None:
        st.error("⚠️ Could not load all models. Please ensure model files are available.")
        st.info("Using sample data for demonstration purposes.")
    
    # Load your actual retail data
    df = load_retail_data()
    
    if df is None:
        st.error("⚠️ Could not load retail_store_inventory.csv. Please ensure the file is in the correct location.")
        st.stop()
    
    # Display data info
    st.sidebar.info(f" Dataset loaded: {len(df)} records")
    st.sidebar.info(f" Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    
    # Product and store selection
    available_products = sorted(df['Product_ID'].unique()) if 'Product_ID' in df.columns else []
    available_stores = sorted(df['Store_ID'].unique()) if 'Store_ID' in df.columns else []
    
    if len(available_products) == 0 or len(available_stores) == 0:
        st.error("⚠️ No products or stores found in the dataset. Please check your data format.")
        st.stop()
    
    # Sidebar controls
    st.sidebar.header(" Forecast Configuration")
    
    selected_product = st.sidebar.selectbox(
        "Select Product:",
        available_products,
        key="product_selector"
    )
    
    selected_store = st.sidebar.selectbox(
        "Select Store:",
        available_stores,
        key="store_selector"
    )
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "Select Model:",
        ["Prophet", "Random Forest", "XGBoost", "Compare All"],
        key="model_selector"
    )
    
    # Forecast period
    forecast_days = st.sidebar.slider(
        "Forecast Days:",
        min_value=7,
        max_value=180,
        value=60,
        step=7
    )
    
    # Generate forecast button
    if st.sidebar.button(" Generate Forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            # Prepare data
            historical_data = df[(df['Product_ID'] == selected_product) & 
                               (df['Store_ID'] == selected_store)].copy()
            
            if len(historical_data) == 0:
                st.error("No data available for selected product and store combination.")
                return
            
            # Make predictions based on model choice
            if model_choice == "Prophet":
                if prophet_model is not None:
                    prophet_data = prepare_prophet_data(df, selected_product, selected_store)
                    future_dates, future_predictions, lower_bound, upper_bound = make_prophet_predictions(
                        prophet_model, prophet_data, forecast_days
                    )
                else:
                    st.error("Prophet model not available.")
                    return
                    
            elif model_choice == "Random Forest":
                if rf_model is not None:
                    ml_data = prepare_features_for_ml(df, selected_product, selected_store)
                    future_dates, future_predictions = make_ml_predictions(
                        rf_model, ml_data, "Random Forest", forecast_days
                    )
                    lower_bound, upper_bound = None, None
                else:
                    st.error("Random Forest model not available.")
                    return
                    
            elif model_choice == "XGBoost":
                if xgb_model is not None:
                    ml_data = prepare_features_for_ml(df, selected_product, selected_store)
                    future_dates, future_predictions = make_ml_predictions(
                        xgb_model, ml_data, "XGBoost", forecast_days
                    )
                    lower_bound, upper_bound = None, None
                else:
                    st.error("XGBoost model not available.")
                    return
                    
            else:  # Compare All
                predictions_dict = {}
                
                # Prophet predictions
                if prophet_model is not None:
                    prophet_data = prepare_prophet_data(df, selected_product, selected_store)
                    dates, preds, _, _ = make_prophet_predictions(prophet_model, prophet_data, forecast_days)
                    if dates is not None:
                        predictions_dict["Prophet"] = preds
                        future_dates = dates
                
                # Random Forest predictions
                if rf_model is not None:
                    ml_data = prepare_features_for_ml(df, selected_product, selected_store)
                    dates, preds = make_ml_predictions(rf_model, ml_data, "Random Forest", forecast_days)
                    if dates is not None:
                        predictions_dict["Random Forest"] = preds
                        if 'future_dates' not in locals():
                            future_dates = dates
                
                # XGBoost predictions
                if xgb_model is not None:
                    ml_data = prepare_features_for_ml(df, selected_product, selected_store)
                    dates, preds = make_ml_predictions(xgb_model, ml_data, "XGBoost", forecast_days)
                    if dates is not None:
                        predictions_dict["XGBoost"] = preds
                        if 'future_dates' not in locals():
                            future_dates = dates
            
            # Store results in session state
            st.session_state.predictions_made = True
            st.session_state.forecast_data = {
                'historical_data': historical_data,
                'future_dates': future_dates,
                'future_predictions': future_predictions if model_choice != "Compare All" else None,
                'lower_bound': lower_bound if model_choice == "Prophet" else None,
                'upper_bound': upper_bound if model_choice == "Prophet" else None,
                'model_choice': model_choice,
                'predictions_dict': predictions_dict if model_choice == "Compare All" else None,
                'selected_product': selected_product,
                'selected_store': selected_store
            }
    
    # Display results
    if st.session_state.predictions_made and st.session_state.forecast_data:
        data = st.session_state.forecast_data
        
        # Key metrics
        st.subheader("📈 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_historical = data['historical_data']['Units_Sold'].mean()
            st.metric("Avg Historical Demand", f"{avg_historical:.1f}", help="Average daily demand from historical data")
        
        with col2:
            if data['model_choice'] != "Compare All":
                avg_forecast = np.mean(data['future_predictions'])
                change = ((avg_forecast - avg_historical) / avg_historical) * 100
                st.metric("Avg Forecast Demand", f"{avg_forecast:.1f}", f"{change:+.1f}%")
            else:
                st.metric("Models Compared", len(data['predictions_dict']), help="Number of models in comparison")
        
        with col3:
            total_historical = data['historical_data']['Units_Sold'].sum()
            st.metric("Total Historical", f"{total_historical:.0f}", help="Total units sold historically")
        
        with col4:
            if data['model_choice'] != "Compare All":
                total_forecast = sum(data['future_predictions'])
                st.metric("Total Forecast", f"{total_forecast:.0f}", help="Total forecasted units")
            else:
                avg_predictions = np.mean([np.mean(preds) for preds in data['predictions_dict'].values()])
                st.metric("Avg Model Prediction", f"{avg_predictions:.1f}", help="Average across all models")
        
        # Main forecast chart
        st.subheader(" Demand Forecast")
        
        if data['model_choice'] == "Compare All":
            # Show model comparison
            comparison_fig = create_model_comparison_chart(data['future_dates'], data['predictions_dict'])
            st.plotly_chart(comparison_fig, use_container_width=True)
            
            # Show individual model charts
            for model_name, predictions in data['predictions_dict'].items():
                individual_fig = create_forecast_chart(
                    data['historical_data'], 
                    data['future_dates'], 
                    predictions,
                    title=f"{model_name} - Demand Forecast"
                )
                st.plotly_chart(individual_fig, use_container_width=True)
        else:
            # Show single model forecast
            forecast_fig = create_forecast_chart(
                data['historical_data'],
                data['future_dates'],
                data['future_predictions'],
                data['lower_bound'],
                data['upper_bound'],
                f"{data['model_choice']} - Demand Forecast"
            )
            st.plotly_chart(forecast_fig, use_container_width=True)
        
        # Inventory analysis
        if data['model_choice'] != "Compare All":
            st.subheader(" Inventory Gap Analysis")
            
            # Generate synthetic inventory levels based on actual data patterns
            if 'Inventory_Level' in data['historical_data'].columns:
                base_inventory = data['historical_data']['Inventory_Level'].mean()
                inventory_std = data['historical_data']['Inventory_Level'].std()
            else:
                base_inventory = data['historical_data']['Units_Sold'].mean() * 1.2
                inventory_std = data['historical_data']['Units_Sold'].std() * 0.5
            
            inventory_levels = np.random.normal(base_inventory, inventory_std, len(data['future_predictions']))
            inventory_levels = np.maximum(inventory_levels, 0)  # Ensure non-negative
            
            inventory_fig = create_inventory_gap_chart(
                data['future_dates'],
                data['future_predictions'],
                inventory_levels
            )
            st.plotly_chart(inventory_fig, use_container_width=True)
            
            # Inventory recommendations
            st.subheader("💡 Inventory Recommendations")
            
            overstocked_days = sum(1 for i, pred in enumerate(data['future_predictions']) 
                                 if inventory_levels[i] > pred * 1.2)
            understocked_days = sum(1 for i, pred in enumerate(data['future_predictions']) 
                                  if inventory_levels[i] < pred * 0.8)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if overstocked_days > 0:
                    st.warning(f"⚠️ **Overstock Risk**: {overstocked_days} days with excess inventory")
                    st.info("💡 Consider reducing order quantities or running promotions")
                else:
                    st.success("✅ No significant overstock risk detected")
            
            with col2:
                if understocked_days > 0:
                    st.error(f"🚨 **Stockout Risk**: {understocked_days} days with insufficient inventory")
                    st.info("💡 Consider increasing order quantities or safety stock")
                else:
                    st.success("✅ No significant stockout risk detected")
        
        # Downloadable forecast data
        st.subheader(" Forecast Data")
        
        if data['model_choice'] != "Compare All":
            forecast_df = pd.DataFrame({
                'Date': data['future_dates'],
                'Forecasted_Demand': data['future_predictions']
            })
            
            if data['lower_bound'] is not None and data['upper_bound'] is not None:
                forecast_df['Lower_Bound'] = data['lower_bound']
                forecast_df['Upper_Bound'] = data['upper_bound']
            
            st.dataframe(forecast_df, use_container_width=True)
            
            # Download button
            csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast Data",
                data=csv,
                file_name=f"forecast_{data['selected_product']}_{data['selected_store']}.csv",
                mime="text/csv"
            )
        else:
            # Show comparison table
            comparison_df = pd.DataFrame({
                'Date': data['future_dates'],
                **data['predictions_dict']
            })
            
            st.dataframe(comparison_df, use_container_width=True)
            
            # Download button
            csv = comparison_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Comparison Data",
                data=csv,
                file_name=f"model_comparison_{data['selected_product']}_{data['selected_store']}.csv",
                mime="text/csv"
            )
    
    else:
        # Welcome message
        st.info("👋 Welcome to the Demand Forecasting Dashboard! Select your parameters and click 'Generate Forecast' to begin.")
        
        # Show actual data preview
        st.subheader("📋 Your Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Dataset summary
        st.subheader(" Dataset Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(df))
            st.metric("Products", len(df['Product_ID'].unique()) if 'Product_ID' in df.columns else 0)
        
        with col2:
            st.metric("Stores", len(df['Store_ID'].unique()) if 'Store_ID' in df.columns else 0)
            if 'Units_Sold' in df.columns:
                st.metric("Avg Daily Sales", f"{df['Units_Sold'].mean():.1f}")
        
        with col3:
            st.metric("Date Range", f"{(df.index.max() - df.index.min()).days} days")
            if 'Category' in df.columns:
                st.metric("Categories", len(df['Category'].unique()))
        
        # Data insights
        st.subheader("📊 Data Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            # Sales trend over time
            if 'Units_Sold' in df.columns:
                daily_sales = df.groupby(df.index.date)['Units_Sold'].sum().reset_index()
                daily_sales.columns = ['Date', 'Total_Sales']
                fig_trend = px.line(daily_sales, x='Date', y='Total_Sales', 
                                  title="Daily Sales Trend")
                st.plotly_chart(fig_trend, use_container_width=True)
            
        with col2:
            # Top products by sales
            if 'Units_Sold' in df.columns and 'Product_ID' in df.columns:
                top_products = df.groupby('Product_ID')['Units_Sold'].sum().sort_values(ascending=False).head(10)
                fig_products = px.bar(x=top_products.index, y=top_products.values, 
                                    title="Top 10 Products by Sales")
                fig_products.update_layout(xaxis_title="Product ID", yaxis_title="Total Sales")
                st.plotly_chart(fig_products, use_container_width=True)

if __name__ == "__main__":
    main()