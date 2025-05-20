import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def create_results_dir(prefix="dart_model_results"):
    """Create a results directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{prefix}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def save_model_params(name, params, metrics=None, results_dir=None):
    """Save model parameters and metrics to a JSON file"""
    if results_dir is None:
        results_dir = "."
    
    data = {
        "model_name": name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": params
    }
    
    if metrics is not None:
        data["metrics"] = metrics
    
    # Save as JSON
    with open(f"{results_dir}/{name}_params.json", 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"✅ Saved {name} parameters to {results_dir}/{name}_params.json")

def create_time_features(df, date_col):
    """Create time-based features from a date column"""
    # Convert to datetime if not already
    if df[date_col].dtype != 'datetime64[ns]':
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Extract date components
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    
    # Create month-year period for grouping
    df['month_year'] = df[date_col].dt.strftime('%Y-%m')
    
    return df

def evaluate_time_series_model(y_true, y_pred, name):
    """Evaluate time series model with various metrics"""
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    # Calculate thresholds
    within_10pct = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10)) < 0.1) * 100
    within_15pct = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10)) < 0.15) * 100
    
    # Print metrics
    print(f"{name} - R²: {r2:.4f}")
    print(f"{name} - RMSE: {rmse:.2f}")
    print(f"{name} - MAE: {mae:.2f}")
    print(f"{name} - MAPE: {mape:.2f}%")
    print(f"{name} - Predictions within 10% of actual: {within_10pct:.2f}%")
    print(f"{name} - Predictions within 15% of actual: {within_15pct:.2f}%")
    
    # Return as dictionary
    return {
        "name": name,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "within_10pct": within_10pct,
        "within_15pct": within_15pct
    }

def plot_time_series_results(df, target_col, pred_col, time_col, output_dir=None):
    """Plot time series results for visualization"""
    # Create plot directory if needed
    if output_dir:
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
    
    # Prepare data
    plot_df = df[[time_col, target_col, pred_col]].copy()
    plot_df = plot_df.sort_values(by=time_col)
    
    # Plot time series
    plt.figure(figsize=(12, 6))
    plt.plot(plot_df[time_col], plot_df[target_col], label='Actual')
    plt.plot(plot_df[time_col], plot_df[pred_col], label='Predicted')
    plt.title(f'Time Series Prediction: {target_col}')
    plt.xlabel(time_col)
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save if output directory provided
    if output_dir:
        plt.savefig(f"{output_dir}/plots/time_series_prediction.png")
        plt.close()
    else:
        plt.show()
    
    # Plot actual vs predicted
    plt.figure(figsize=(8, 8))
    plt.scatter(plot_df[target_col], plot_df[pred_col], alpha=0.5)
    plt.plot([plot_df[target_col].min(), plot_df[target_col].max()], 
              [plot_df[target_col].min(), plot_df[target_col].max()], 'r--')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True)
    plt.tight_layout()
    
    # Save if output directory provided
    if output_dir:
        plt.savefig(f"{output_dir}/plots/actual_vs_predicted.png")
        plt.close()
    else:
        plt.show()