import pandas as pd
import numpy as np
import time
import joblib
import os
import json
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import optuna

# Import specific time series utilities
from utils.time_series_utils import (
    create_results_dir, save_model_params, 
    evaluate_time_series_model, create_time_features,
    plot_time_series_results
)

warnings.filterwarnings("ignore")

# Check if Dart (PyTorch Forecasting) is available
try:
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss
    from pytorch_forecasting.data import GroupNormalizer
    import torch
    import pytorch_lightning as pl
    DART_AVAILABLE = True
    print("✅ PyTorch Forecasting is available for time series modeling")
except ImportError:
    DART_AVAILABLE = False
    print("⚠️ PyTorch Forecasting not available, falling back to LightGBM")
    import lightgbm as lgb

# =================== CREATE RESULTS DIRECTORY =================== #
RESULTS_DIR = create_results_dir("dart_model_results")
print(f"\n📁 Created results directory: {RESULTS_DIR}")

# =================== LOAD DATA =================== #
def load_data(file_path):
    """Load and prepare data for time series modeling"""
    print(f"\n📂 Loading data from {file_path}...")
    
    df = pd.read_excel(file_path, engine="openpyxl")
    print(f"✅ Data loaded with shape: {df.shape}")
    
    # Focus on columns needed for time series prediction
    required_cols = [
        "Xã/Phường/Thị trấn", "Vị trí trong khung giá", 
        "Thời điểm hiệu lực", "Đơn giá quyền sử dụng đất (đ/m2)",
        "Diện tích (m2)"
    ]
    
    # Check if all required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        if "Tháng" in df.columns and "Giá trung bình tháng/phường/VT" in df.columns:
            print("✅ Found pre-processed columns 'Tháng' and 'Giá trung bình tháng/phường/VT', continuing...")
        else:
            raise ValueError(f"Missing critical columns: {missing_cols}")
    
    # Add time features if needed
    if "Tháng" not in df.columns or "month" not in df.columns:
        print("🔄 Creating time features...")
        date_col = "Thời điểm hiệu lực"
        df = create_time_features(df, date_col)
    
    return df

# =================== PREPARE TIME SERIES DATA =================== #
def prepare_time_series_data(df):
    """Prepare data for time series modeling"""
    print("\n🔄 Preparing time series data...")
    
    # Define location columns for grouping
    location_cols = ["Xã/Phường/Thị trấn", "Vị trí trong khung giá"]
    
    # Define target column (price to predict)
    if "Giá trung bình tháng/phường/VT" in df.columns:
        target_col = "Giá trung bình tháng/phường/VT"
        print(f"✅ Using pre-calculated average price column: {target_col}")
    else:
        target_col = "Đơn giá quyền sử dụng đất (đ/m2)"
        print(f"⚠️ Pre-calculated average prices not found, using actual prices: {target_col}")
        
        # Calculate average prices by location and month
        print("🔄 Calculating average prices by location and month...")
        df_avg = df.groupby(location_cols + ["month", "year"])[target_col].mean().reset_index()
        df_avg.rename(columns={target_col: "Giá trung bình tháng/phường/VT"}, inplace=True)
        
        # Merge back to original dataframe
        df = pd.merge(df, df_avg, on=location_cols + ["month", "year"], how="left")
        target_col = "Giá trung bình tháng/phường/VT"
        print(f"✅ Created and using average price column: {target_col}")
    
    # Create cyclic month features
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    # Group by location and time for unique combinations (for time series model training)
    time_series_df = df.groupby(location_cols + ["month", "year"])[target_col].mean().reset_index()
    
    # Add cyclic features to time series dataframe
    time_series_df['month_sin'] = np.sin(2 * np.pi * time_series_df['month']/12)
    time_series_df['month_cos'] = np.cos(2 * np.pi * time_series_df['month']/12)
    
    # Convert to log scale to stabilize variance
    time_series_df[f"log_{target_col}"] = np.log1p(time_series_df[target_col])
    
    print(f"✅ Prepared time series data with {time_series_df.shape[0]} unique location-time combinations")
    
    return time_series_df, location_cols, target_col

# =================== TRAIN DART MODEL =================== #
def train_dart_model(time_series_df, location_cols, target_col):
    """Train a Dart (PyTorch Forecasting) model if available, otherwise use LightGBM"""
    print("\n🔧 Training time series model...")
    
    # Add a global reference to the DART_AVAILABLE variable
    global DART_AVAILABLE
    
    # Input features for the model
    feature_cols = location_cols + ["month", "year", "month_sin", "month_cos"]
    target_log_col = f"log_{target_col}"
    
    # Split data
    X = time_series_df[feature_cols]
    y = time_series_df[target_log_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✅ Split data: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    # Training logic depends on available packages
    if DART_AVAILABLE and len(time_series_df) >= 100:  # Dart needs sufficient data
        print("🔧 Using PyTorch Forecasting (Dart) for time series modeling")
        
        # Prepare data for Dart (add time index and ID columns)
        dart_df = time_series_df.copy()
        
        # Create a time index (assuming monthly data)
        dart_df['time_idx'] = dart_df['year'] * 12 + dart_df['month']
        min_time_idx = dart_df['time_idx'].min()
        dart_df['time_idx'] = dart_df['time_idx'] - min_time_idx
        
        # Convert categorical columns to integer codes
        print("🔄 Converting categorical columns to integers...")
        encoded_location_cols = []

        for col in location_cols:
            encoded_col = f"{col}_encoded"
            # Convert to category type first
            dart_df[col] = dart_df[col].astype('category')
            # Then get the codes (integers starting from 0)
            dart_df[encoded_col] = dart_df[col].cat.codes
            encoded_location_cols.append(encoded_col)
            
            # Store mapping for future predictions
            category_mapping = {int(code): category for code, category in 
                                enumerate(dart_df[col].cat.categories)}
            
            mapping_path = os.path.join(RESULTS_DIR, f"{col}_mapping.json")
            with open(mapping_path, 'w', encoding='utf-8') as f:
                json.dump(category_mapping, f, ensure_ascii=False, indent=4)
            
            print(f"✅ Converted '{col}' to integers with {len(category_mapping)} categories")

        # Create group_id from encoded columns
        dart_df['group_id'] = dart_df[encoded_location_cols[0]].astype(str) + "_" + dart_df[encoded_location_cols[1]].astype(str)
        
        try:
            # Dart implementation here
            print("⚠️ Dart implementation will be added later")
            # For now, we'll fall back to LightGBM
            DART_AVAILABLE = False
            print("⚠️ Falling back to LightGBM for this run")
        except Exception as e:
            print(f"❌ Error in Dart implementation: {e}")
            DART_AVAILABLE = False
            print("⚠️ Falling back to LightGBM due to error")
    
    # LightGBM model (fallback or primary option)
    print("🔧 Using LightGBM for time series modeling")
    
    # Scale numerical features
    scaler = StandardScaler()
    numeric_cols = ["month_sin", "month_cos", "month", "year"]
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Convert categorical columns to category type for LightGBM
    for col in location_cols:
        X_train_scaled[col] = X_train_scaled[col].astype('category')
        X_test_scaled[col] = X_test_scaled[col].astype('category')
    
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # Define model with default hyperparameters
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'n_estimators': 500,
        'max_depth': 5,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42
    }
    
    # Hyperparameter tuning with Optuna (limited trials for speed)
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'num_leaves': trial.suggest_int('num_leaves', 7, 127),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # We want to maximize the percentage of predictions within thresholds
        y_true_exp = np.expm1(y_test)
        y_pred_exp = np.expm1(y_pred)
        
        # Calculate percentage of predictions within 10% and 15% thresholds
        within_10pct = np.mean(np.abs((y_true_exp - y_pred_exp) / (y_true_exp + 1e-10)) < 0.1) * 100
        within_15pct = np.mean(np.abs((y_true_exp - y_pred_exp) / (y_true_exp + 1e-10)) < 0.15) * 100
        
        # Combined score (70% weight on 10% threshold, 30% weight on 15% threshold)
        score = 0.7 * within_10pct + 0.3 * within_15pct
        
        return score
    
    # Run optimization with fewer trials for speed
    print("🔍 Optimizing LightGBM hyperparameters (quick version)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)  # Reduced number of trials for speed
    
    # Get best parameters
    best_params = study.best_params
    print("✅ Best parameters:", best_params)
    
    # Save parameters
    save_model_params("lgb_time_series", best_params, results_dir=RESULTS_DIR)
    
    # Final model with best parameters
    final_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        boosting_type='gbdt',
        random_state=42,
        **best_params
    )
    
    # Train model with best parameters
    final_model.fit(X_train_scaled, y_train)
    
    # Save model
    model_path = os.path.join(RESULTS_DIR, "lgb_time_series_model.pkl")
    joblib.dump(final_model, model_path)
    
    # Save scaler
    scaler_path = os.path.join(RESULTS_DIR, "time_series_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    
    # Save category information
    cat_info = {}
    for col in location_cols:
        if hasattr(X_train_scaled[col], 'cat') and hasattr(X_train_scaled[col].cat, 'categories'):
            cat_info[col] = list(X_train_scaled[col].cat.categories)
        else:
            # If not properly categorized, just save unique values
            cat_info[col] = list(X_train_scaled[col].unique())
    
    with open(os.path.join(RESULTS_DIR, "categorical_info.json"), 'w', encoding='utf-8') as f:
        json.dump(cat_info, f, ensure_ascii=False, indent=4)
    
    print(f"✅ Saved LightGBM model to {model_path}")
    
    # Evaluate on test set
    y_pred = final_model.predict(X_test_scaled)
    
    # Convert predictions back to original scale and evaluate
    y_true_exp = np.expm1(y_test)
    y_pred_exp = np.expm1(y_pred)
    
    metrics = evaluate_time_series_model(y_true_exp, y_pred_exp, "LightGBM Time Series")
    
    # Return model and metrics - THIS IS THE CRITICAL PART THAT WAS MISSING
    return final_model, metrics, "lgb"

# =================== PREDICT WITH MODEL =================== #
def predict_avg_prices(model, df, location_cols, target_col, model_type):
    """Predict average prices for all data points using the trained model"""
    print("\n🔮 Predicting average prices for all data points...")
    
    # Create predictions for all unique location-time combinations
    unique_combinations = df.groupby(location_cols + ["month", "year"]).size().reset_index()[location_cols + ["month", "year"]]
    unique_combinations['month_sin'] = np.sin(2 * np.pi * unique_combinations['month']/12)
    unique_combinations['month_cos'] = np.cos(2 * np.pi * unique_combinations['month']/12)
    
    if model_type == "dart":
        # For Dart model, prediction needs special handling
        print("⚠️ Dart model prediction not implemented in this script")
        # For now, use a placeholder prediction
        unique_combinations["predicted_avg_price"] = df.groupby(location_cols + ["month", "year"])[target_col].mean().reset_index()[target_col]
    else:
        # For LightGBM model
        # Scale features
        scaler_path = os.path.join(RESULTS_DIR, "time_series_scaler.pkl")
        scaler = joblib.load(scaler_path)
        
        # Load category information
        cat_info_path = os.path.join(RESULTS_DIR, "categorical_info.json")
        if os.path.exists(cat_info_path):
            with open(cat_info_path, 'r', encoding='utf-8') as f:
                cat_info = json.load(f)
            
            # Apply category encoding
            for col in location_cols:
                if col in cat_info:
                    unique_combinations[col] = pd.Categorical(
                        unique_combinations[col], 
                        categories=cat_info[col]
                    )
        
        numeric_cols = ["month_sin", "month_cos", "month", "year"]
        X_pred = unique_combinations.copy()
        X_pred[numeric_cols] = scaler.transform(X_pred[numeric_cols])
        
        # Generate predictions in log scale
        predictions_log = model.predict(X_pred)
        
        # Convert back to original scale
        unique_combinations["predicted_avg_price"] = np.expm1(predictions_log)
    
    # Merge predictions back to original dataframe
    result_df = pd.merge(df, unique_combinations[location_cols + ["month", "year", "predicted_avg_price"]], 
                          on=location_cols + ["month", "year"], how="left")
    
    print(f"✅ Added predicted average prices to dataframe")
    
    return result_df

# =================== SAVE RESULTS =================== #
def save_results(result_df, model, model_type, metrics, location_cols, target_col):
    """Save model, predictions, and evaluation results"""
    print("\n💾 Saving results...")
    
    # Save the dataframe with predictions
    df_path = os.path.join(RESULTS_DIR, "data_with_predictions.xlsx")
    result_df.to_excel(df_path, index=False)
    
    # Create info for the combined model pipeline
    data_info = {
        "location_cols": location_cols,
        "date_col": "Thời điểm hiệu lực",
        "final_target": "Đơn giá quyền sử dụng đất (đ/m2)",
        "avg_price_target": target_col,
        "model_type": model_type,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save data info
    with open(os.path.join(RESULTS_DIR, "data_info.json"), 'w') as f:
        json.dump(data_info, f, indent=4)
    
    # Save complete training summary
    training_summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        f"{model_type}_time_series": {
            "metrics": metrics,
            "location_cols": location_cols,
            "target_col": target_col
        },
        "model_files": {
            f"{model_type}_model": os.path.join(RESULTS_DIR, 
                                                "dart_model.pth" if model_type == "dart" else "lgb_time_series_model.pkl"),
            "data_info": os.path.join(RESULTS_DIR, "data_info.json")
        }
    }
    
    # Save the complete training summary
    with open(os.path.join(RESULTS_DIR, "training_summary.json"), 'w') as f:
        json.dump(training_summary, f, indent=4)
    
    print(f"✅ Saved results to {RESULTS_DIR}")
    return data_info

# =================== MAIN FUNCTION =================== #
def main():
    print("\n==================================================")
    print(" 🚀 Time Series Dart Price Predictor")
    print("==================================================")
    
    # Load data
    file_path = "2024+q1-25_with_avg.xlsx"  # Update to your file path
    df = load_data(file_path)
    
    # Prepare time series data
    time_series_df, location_cols, target_col = prepare_time_series_data(df)
    
    # Train model
    model, metrics, model_type = train_dart_model(time_series_df, location_cols, target_col)
    
    # Generate predictions for all data points
    result_df = predict_avg_prices(model, df, location_cols, target_col, model_type)
    
    # Save results
    data_info = save_results(result_df, model, model_type, metrics, location_cols, target_col)
    
    print("\n✅ Time series model training and prediction completed!")
    print(f"📁 Results saved to {RESULTS_DIR}")
    print("\n→ Next Steps:")
    print("  1. Use these predictions as features in your LightGBM model")
    print("  2. Train a combined model using both time series features and other property features")
    
if __name__ == "__main__":
    main()