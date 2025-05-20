import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
import os
import sys
import warnings
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore")

# =================== CONFIGURATION =================== #
# Set this to the folder containing your combined model results
RESULTS_DIR = 'combined_model_results_20250520_101549'  # Update this to your timestamp folder
# File path for test data
TEST_FILE_PATH = 'T4_hn_final2.xlsx'  # Change to your test file

# =================== WRAPPER ENCODER =================== #
class TargetEncodingWrapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = TargetEncoder()
        
    def fit(self, X, y=None):
        self.encoder.fit(X, y)
        return self
    
    def transform(self, X):
        return self.encoder.transform(X).values

# =================== CREATE EVALUATION DIRECTORY =================== #
def create_eval_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = f"combined_model_evaluation_{timestamp}"
    os.makedirs(eval_dir, exist_ok=True)
    return eval_dir

EVAL_DIR = create_eval_dir()
print(f"\n📁 Created evaluation results directory: {EVAL_DIR}")

# =================== LOAD MODELS =================== #
def load_models(results_dir):
    """Load average price model and final price model"""
    try:
        # Load training summary to get file paths
        with open(f"{results_dir}/training_summary.json", 'r') as f:
            training_summary = json.load(f)
        print("✅ Loaded training summary")
        
        # Extract file paths
        avg_price_model_path = training_summary["model_files"]["avg_price_model"]
        lightgbm_model_path = training_summary["model_files"]["lightgbm_model"]
        x_cols_path = training_summary["model_files"]["X_train_columns"]
        data_info_path = training_summary["model_files"]["data_info"]
        
        # Load models
        avg_price_model = joblib.load(avg_price_model_path)
        print(f"✅ Loaded average price model from {avg_price_model_path}")
        
        lightgbm_model = joblib.load(lightgbm_model_path)
        print(f"✅ Loaded LightGBM model from {lightgbm_model_path}")
        
        # Load column information and data info
        X_columns = joblib.load(x_cols_path)
        print(f"✅ Loaded column information from {x_cols_path}")
        
        with open(data_info_path, 'r') as f:
            data_info = json.load(f)
        print(f"✅ Loaded data info from {data_info_path}")
        
        # Extract columns required for the average price model
        avg_price_cols_path = os.path.join(os.path.dirname(avg_price_model_path), "X_avg_columns.pkl")
        if os.path.exists(avg_price_cols_path):
            avg_price_columns = joblib.load(avg_price_cols_path)
            print(f"✅ Loaded average price model columns from {avg_price_cols_path}")
        else:
            # If columns file not found, extract from data_info
            location_cols = data_info["location_cols"]
            avg_price_columns = location_cols + ["month", "year", "month_sin", "month_cos"]
            print("⚠️ Couldn't find average price model columns file, using default columns")
            
        return {
            "avg_price_model": avg_price_model,
            "lightgbm_model": lightgbm_model,
            "X_columns": X_columns,
            "avg_price_columns": avg_price_columns,
            "data_info": data_info
        }
    
    except FileNotFoundError as e:
        print(f"❌ Error loading models: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

# =================== LOAD DATA =================== #
def load_and_preprocess_data(file_path, data_info):
    """Load and preprocess new data"""
    print(f"\n📊 Loading test data from {file_path}...")
    
    try:
        # Load data
        df = pd.read_excel(file_path, engine="openpyxl")
        print(f"✅ Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Extract needed information
        date_col = data_info.get("date_col", "Thời điểm hiệu lực")
        target_col = data_info.get("final_target", "Đơn giá quyền sử dụng đất (đ/m2)")
        
        # Check if required columns exist
        missing_cols = []
        required_cols = [date_col, target_col]
        required_cols.extend(data_info.get("location_cols", []))
        
        for col in required_cols:
            if col not in df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            sys.exit(1)
        
        # Process date column - FIXED for DD/MM/YYYY format
        if df[date_col].dtype == 'object':
            try:
                # Try parsing with dayfirst=True for DD/MM/YYYY format
                df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
                print(f"✅ Parsed dates with DD/MM/YYYY format: {df[date_col].iloc[0]}")
            except Exception as e:
                print(f"⚠️ Warning with date parsing: {e}")
                # If parsing fails, try a more explicit approach
                df[date_col] = pd.to_datetime(df[date_col], format='%d/%m/%Y', errors='coerce')
                print(f"✅ Parsed dates with explicit DD/MM/YYYY format: {df[date_col].iloc[0]}")
        
        # Check for NaT values after date parsing
        if df[date_col].isna().any():
            print(f"⚠️ Warning: {df[date_col].isna().sum()} dates could not be parsed")
            print(f"⚠️ Example unparsed date: {df.loc[df[date_col].isna(), date_col].iloc[0] if df[date_col].isna().any() else 'None'}")
        
        # Create time features
        df['month'] = df[date_col].dt.month
        df['year'] = df[date_col].dt.year
        df['quarter'] = df[date_col].dt.quarter
        df['month_year'] = df[date_col].dt.strftime('%Y-%m')
        
        # Create cyclic features
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        # Create log_dientich feature if Diện tích exists
        if 'Diện tích (m2)' in df.columns and 'log_dientich' not in df.columns:
            df['log_dientich'] = np.log1p(df['Diện tích (m2)'])
            print("✅ Created log_dientich feature")
        
        # Add "Tháng" column if it doesn't exist
        if 'Tháng' not in df.columns and 'month' in df.columns:
            df['Tháng'] = df['month']
            print("✅ Created Tháng feature from month")
        
        # Log transform target for output
        df['log_target'] = np.log1p(df[target_col])
        
        # Handle missing values
        df = df.dropna(subset=[target_col])
        
        # Handle missing dates after parsing
        df = df.dropna(subset=[date_col])
        
        print(f"✅ Preprocessed data with final shape: {df.shape}")
        return df
    
    except Exception as e:
        print(f"❌ Error preprocessing data: {e}")
        sys.exit(1)

# =================== PREDICT AVERAGE PRICE =================== #
def predict_average_price(df, avg_price_model, avg_price_columns, data_info):
    """Predict average price using the average price model"""
    print("\n🔮 Predicting average prices...")
    
    try:
        # Prepare features for the average price model
        X_avg = df[avg_price_columns].copy()
        
        # Make predictions
        avg_price_log = avg_price_model.predict(X_avg)
        avg_price = np.expm1(avg_price_log)
        
        # Add to dataframe
        df['predicted_avg_price'] = avg_price
        
        # Add the result as the "Giá trung bình tháng/phường/VT" column
        df['Giá trung bình tháng/phường/VT'] = avg_price
        
        # Calculate ratio if target exists
        final_target_col = data_info.get("final_target", "Đơn giá quyền sử dụng đất (đ/m2)")
        if final_target_col in df.columns:
            df['price_to_avg_ratio'] = df[final_target_col] / df['predicted_avg_price']
        
        print(f"✅ Added predicted average price to dataframe")
        return df
    
    except Exception as e:
        print(f"❌ Error predicting average prices: {e}")
        sys.exit(1)

# =================== PREDICT FINAL PRICE =================== #
def predict_final_price(df, lightgbm_model, X_columns, data_info):
    """Predict final price using the LightGBM model"""
    print("\n🔮 Predicting final prices...")
    
    try:
        # Create a copy of the dataframe for prediction
        pred_df = df.copy()
        
        # Make a list of columns that exist in both X_columns and pred_df
        available_cols = [col for col in X_columns if col in pred_df.columns]
        missing_cols = [col for col in X_columns if col not in pred_df.columns]
        
        if missing_cols:
            print(f"⚠️ Missing columns for LightGBM model: {missing_cols}")
            # Try to create missing columns
            for col in missing_cols:
                if col == 'log_dientich' and 'Diện tích (m2)' in pred_df.columns:
                    pred_df['log_dientich'] = np.log1p(pred_df['Diện tích (m2)'])
                    available_cols.append('log_dientich')
                    print(f"✅ Created missing column: log_dientich")
                elif col == 'Tháng' and 'month' in pred_df.columns:
                    pred_df['Tháng'] = pred_df['month']
                    available_cols.append('Tháng')
                    print(f"✅ Created missing column: Tháng")
                elif col == 'Giá trung bình tháng/phường/VT' and 'predicted_avg_price' in pred_df.columns:
                    pred_df['Giá trung bình tháng/phường/VT'] = pred_df['predicted_avg_price']
                    available_cols.append('Giá trung bình tháng/phường/VT')
                    print(f"✅ Created missing column: Giá trung bình tháng/phường/VT from predicted_avg_price")
                else:
                    print(f"⚠️ Cannot create missing column: {col}")
        
        # Check if we have all the required columns
        still_missing = [col for col in X_columns if col not in available_cols]
        if still_missing:
            print(f"❌ Still missing columns for LightGBM model: {still_missing}")
            raise ValueError(f"Missing required columns for prediction: {still_missing}")
        
        # Extract features for prediction
        X = pred_df[available_cols]
        
        # Make predictions
        target_col = data_info.get("final_target", "Đơn giá quyền sử dụng đất (đ/m2)")
        y_pred_log = lightgbm_model.predict(X)
        y_pred = np.expm1(y_pred_log)
        
        # Add predictions to original dataframe
        df['predicted_price'] = y_pred
        
        print(f"✅ Added predicted final price to dataframe")
        return df, y_pred
    
    except Exception as e:
        print(f"❌ Error predicting final prices: {str(e)}")
        sys.exit(1)

# =================== EVALUATION FUNCTION =================== #
def evaluate_predictions(df, data_info):
    """Evaluate predictions against actual values"""
    print("\n📊 Evaluating predictions...")
    
    try:
        target_col = data_info.get("final_target", "Đơn giá quyền sử dụng đất (đ/m2)")
        
        # Check if target exists for evaluation
        if target_col not in df.columns:
            print("⚠️ Target column not found in test data, cannot evaluate predictions")
            return None
        
        # Get actual and predicted values
        y_true = df[target_col].values
        y_pred = df['predicted_price'].values
        
        # Calculate metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        # Calculate APE thresholds
        within_5pct = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10)) < 0.05) * 100
        within_10pct = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10)) < 0.1) * 100
        within_15pct = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10)) < 0.15) * 100
        
        # Print metrics
        print(f"\nModel Performance Metrics:")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"Predictions within 5% of actual: {within_5pct:.2f}%")
        print(f"Predictions within 10% of actual: {within_10pct:.2f}%")
        print(f"Predictions within 15% of actual: {within_15pct:.2f}%")
        
        # Create visualization
        create_evaluation_plots(df, target_col, EVAL_DIR)
        
        # Return metrics
        metrics = {
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "within_5pct": within_5pct,
            "within_10pct": within_10pct,
            "within_15pct": within_15pct
        }
        
        # Save metrics to JSON
        with open(f"{EVAL_DIR}/metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Save results to Excel
        df_results = df.copy()
        df_results['absolute_error'] = np.abs(y_true - y_pred)
        df_results['percentage_error'] = np.abs((y_true - y_pred) / (y_true + 1e-10)) * 100
        df_results['within_10pct'] = df_results['percentage_error'] < 10
        df_results['within_15pct'] = df_results['percentage_error'] < 15
        
        df_results.to_excel(f"{EVAL_DIR}/prediction_results.xlsx", index=False)
        
        print(f"✅ Saved evaluation metrics to {EVAL_DIR}/metrics.json")
        print(f"✅ Saved detailed results to {EVAL_DIR}/prediction_results.xlsx")
        
        return metrics
    
    except Exception as e:
        print(f"❌ Error evaluating predictions: {e}")
        return None

# =================== CREATE EVALUATION PLOTS =================== #
def create_evaluation_plots(df, target_col, output_dir):
    """Create evaluation plots"""
    print("\n📊 Creating evaluation plots...")
    
    try:
        # Actual vs Predicted
        plt.figure(figsize=(10, 8))
        plt.scatter(df[target_col], df['predicted_price'], alpha=0.5)
        plt.plot([df[target_col].min(), df[target_col].max()], 
                 [df[target_col].min(), df[target_col].max()], 'r--')
        plt.title('Actual vs Predicted Prices')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/actual_vs_predicted.png")
        plt.close()
        
        # Residuals plot
        residuals = df[target_col] - df['predicted_price']
        plt.figure(figsize=(10, 6))
        plt.scatter(df['predicted_price'], residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residuals vs Predicted Values')
        plt.xlabel('Predicted Price')
        plt.ylabel('Residuals')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/residuals.png")
        plt.close()
        
        # Histogram of errors
        ape = np.abs((df[target_col] - df['predicted_price']) / (df[target_col] + 1e-10)) * 100
        plt.figure(figsize=(10, 6))
        plt.hist(ape, bins=50, alpha=0.75)
        plt.axvline(x=10, color='r', linestyle='--', label='10% threshold')
        plt.axvline(x=15, color='g', linestyle='--', label='15% threshold')
        plt.title('Distribution of Absolute Percentage Error')
        plt.xlabel('Absolute Percentage Error (%)')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ape_distribution.png")
        plt.close()
        
        print(f"✅ Saved evaluation plots to {output_dir}")
    
    except Exception as e:
        print(f"❌ Error creating evaluation plots: {e}")

# =================== MAIN FUNCTION =================== #
def main():
    print("\n==================================================")
    print(" 🚀 Combined Model Test Script")
    print("==================================================")
    
    # Load models and metadata
    print("\n🔍 Loading models and metadata...")
    models_data = load_models(RESULTS_DIR)
    
    # Load and preprocess test data
    print("\n📊 Loading test data...")
    df = load_and_preprocess_data(TEST_FILE_PATH, models_data["data_info"])
    
    # Predict average price
    df = predict_average_price(
        df, 
        models_data["avg_price_model"], 
        models_data["avg_price_columns"], 
        models_data["data_info"]
    )
    
    # Predict final price
    df, _ = predict_final_price(
        df, 
        models_data["lightgbm_model"], 
        models_data["X_columns"], 
        models_data["data_info"]
    )
    
    # Evaluate predictions
    evaluate_predictions(df, models_data["data_info"])
    
    print("\n✅ Testing completed!")
    print(f"📁 Results saved to {EVAL_DIR}")

if __name__ == "__main__":
    main()