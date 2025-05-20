import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import time
import warnings
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import TargetEncoder
import statsmodels.api as sm

# Import utility functions
from utils.time_series_utils import (
    create_results_dir, save_model_params, 
    evaluate_time_series_model, create_time_features
)

warnings.filterwarnings("ignore")

# =================== CHECK GPU AVAILABILITY =================== #
print("\n🔍 Checking GPU availability for LightGBM...")
use_gpu = False
try:
    if lgb.GPUTreeLearner:
        use_gpu = True
        print("✅ GPU is available and will be used for training")
except Exception:
    print("⚠️ GPU not available, using CPU for training")

# =================== CREATE RESULTS DIRECTORY =================== #
RESULTS_DIR = create_results_dir("combined_model_results")
print(f"\n📁 Created results directory: {RESULTS_DIR}")

# =================== WRAPPER ENCODER =================== #
class TargetEncodingWrapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = TargetEncoder()
        
    def fit(self, X, y=None):
        self.encoder.fit(X, y)
        return self
    
    def transform(self, X):
        return self.encoder.transform(X).values
    
# =================== CUSTOM MAPE SCORER =================== #
def mape_score(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

# Create a scorer for MAPE
from sklearn.metrics import make_scorer
mape_scorer = make_scorer(mape_score, greater_is_better=False)

# =================== CUSTOM APE THRESHOLD SCORER =================== #
def ape_threshold_score(y_true, y_pred, threshold=0.1):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10)) < threshold) * 100

# Create scorers for APE thresholds
ape_under_10_scorer = make_scorer(
    lambda y_true, y_pred: ape_threshold_score(y_true, y_pred, threshold=0.1),
    greater_is_better=True)

ape_under_15_scorer = make_scorer(
    lambda y_true, y_pred: ape_threshold_score(y_true, y_pred, threshold=0.15),
    greater_is_better=True)

# Combined scorer
def combined_threshold_score(y_true, y_pred):
    under_10 = ape_threshold_score(y_true, y_pred, threshold=0.1)
    under_15 = ape_threshold_score(y_true, y_pred, threshold=0.15)
    return under_10 * 0.7 + under_15 * 0.3

combined_threshold_scorer = make_scorer(combined_threshold_score, greater_is_better=True)

# =================== LOAD DATA =================== #
print("\n📂 Loading data...")
start_time = time.time()
file_path = "2024+q1-25_with_avg.xlsx"  # Using the file with avg prices from FeatureEngineering.py
df = pd.read_excel(file_path, engine="openpyxl")

columns_needed = [
    "Thành phố/Quận/Huyện/Thị xã", "Xã/Phường/Thị trấn", "Đường phố",
    "Diện tích (m2)", "Kích thước mặt tiền (m)", "Kích thước chiều dài", "Số mặt tiền tiếp giáp",
    "Độ rộng ngõ/ngách nhỏ nhất (Từ đường chính đến BĐS)", "Khoảng cách đến đường chính (m)",
    "Vị trí trong khung giá", "Lợi thế kinh doanh", "Đơn giá trung bình xã/phường theo VT", 
    "Thời điểm hiệu lực", "Đơn giá quyền sử dụng đất (đ/m2)", "Đơn giá trung bình đường phố theo VT",
    "Tháng", "Giá trung bình tháng/phường/VT"  # New columns from feature engineering
]

# Filter required columns
if set(columns_needed).issubset(df.columns):
    df = df[columns_needed].copy()
    print(f"✅ Data loaded in {time.time() - start_time:.2f} seconds")
    print(f"📊 Initial dataset shape: {df.shape}")
else:
    missing_cols = set(columns_needed) - set(df.columns)
    print(f"❌ Missing columns: {missing_cols}")
    # Try to continue with available columns
    available_cols = list(set(columns_needed).intersection(set(df.columns)))
    df = df[available_cols].copy()
    print(f"⚠️ Continuing with available columns. Dataset shape: {df.shape}")

# =================== NaN HANDLING =================== #
print("\n🔍 Checking for NaN values...")
# Handle NaN values
df = df.dropna()
print(f"📊 Clean dataset shape: {df.shape}")

# =================== DATE HANDLING =================== #
print("\n🔄 Processing date column...")
# Add time-based features (making sure we use the right date column name)
date_col = "Thời điểm hiệu lực"
df = create_time_features(df, date_col)

# Define column categories
cat_cols = [
    "Thành phố/Quận/Huyện/Thị xã", "Xã/Phường/Thị trấn", "Đường phố",
    "Vị trí trong khung giá", "Lợi thế kinh doanh"
]

# Remaining columns (excluding target and non-feature columns)
num_cols = [col for col in df.columns if col not in cat_cols + 
           ["Đơn giá quyền sử dụng đất (đ/m2)", "Thời điểm hiệu lực", "month_year"]]

# =================== TRAIN AVERAGE PRICE MODEL =================== #
print("\n🔄 Building model to predict average price by location and time...")

# Define variables
time_series_target = "Giá trung bình tháng/phường/VT"  # Target is the average price
location_cols = ["Xã/Phường/Thị trấn", "Vị trí trong khung giá"]
final_target_col = "Đơn giá quyền sử dụng đất (đ/m2)"

# Create additional time features
df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
df['month_cos'] = np.cos(2 * np.pi * df['month']/12)

# 1. First, train a model to predict the average price by location and time
# Group by location and month to get unique combinations
avg_price_df = df.groupby(location_cols + ["month", "year"])[time_series_target].mean().reset_index()

# Create cyclic month features for time series model
avg_price_df['month_sin'] = np.sin(2 * np.pi * avg_price_df['month']/12)
avg_price_df['month_cos'] = np.cos(2 * np.pi * avg_price_df['month']/12)

# Define features for average price prediction
avg_price_features = location_cols + ["month", "year", "month_sin", "month_cos"]

# Split data for the average price model
X_avg = avg_price_df[avg_price_features]
y_avg = np.log1p(avg_price_df[time_series_target])  # Log transform to stabilize variance

X_avg_train, X_avg_test, y_avg_train, y_avg_test = train_test_split(
    X_avg, y_avg, test_size=0.2, random_state=42
)

# Create a pipeline for the average price model
avg_price_preprocessor = ColumnTransformer([
    ("num", StandardScaler(), ["month", "year", "month_sin", "month_cos"]),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), location_cols),
], remainder='drop')

# LightGBM model for average price prediction
avg_price_pipeline = Pipeline([
    ("preprocessor", avg_price_preprocessor),
    ("regressor", lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
    ))
])

# Train the average price model
print("🔧 Training average price prediction model...")
avg_price_pipeline.fit(X_avg_train, y_avg_train)

# Evaluate the average price model
avg_price_train_pred = avg_price_pipeline.predict(X_avg_train)
avg_price_test_pred = avg_price_pipeline.predict(X_avg_test)

# Convert predictions back to original scale
avg_price_train_pred_exp = np.expm1(avg_price_train_pred)
avg_price_test_pred_exp = np.expm1(avg_price_test_pred)
y_avg_train_exp = np.expm1(y_avg_train)
y_avg_test_exp = np.expm1(y_avg_test)

# Calculate metrics for average price model
print("\n📊 Average Price Model Performance:")
avg_price_train_metrics = evaluate_time_series_model(
    y_avg_train_exp, avg_price_train_pred_exp, "Average Price Model (Train)"
)
avg_price_test_metrics = evaluate_time_series_model(
    y_avg_test_exp, avg_price_test_pred_exp, "Average Price Model (Test)"
)

# Save the average price model
avg_price_model_dir = os.path.join(RESULTS_DIR, "avg_price_model")
os.makedirs(avg_price_model_dir, exist_ok=True)
joblib.dump(avg_price_pipeline, os.path.join(avg_price_model_dir, "avg_price_model.pkl"))
joblib.dump(X_avg.columns.tolist(), os.path.join(avg_price_model_dir, "X_avg_columns.pkl"))

# =================== PREPARE FINAL DATASET FOR LIGHTGBM =================== #
print("\n🔄 Preparing data for final LightGBM model...")

# Generate predicted average prices for all data (including test data)
avg_price_features_df = df[location_cols + ["month", "year"]].copy()
avg_price_features_df['month_sin'] = df['month_sin']
avg_price_features_df['month_cos'] = df['month_cos']

# Use the trained avg price model to predict for all data
predicted_avg_price_log = avg_price_pipeline.predict(avg_price_features_df)
predicted_avg_price = np.expm1(predicted_avg_price_log)

# Add the predicted average price as a feature
df['predicted_avg_price'] = predicted_avg_price

# Create ratio feature (price to predicted average price ratio)
df['price_to_avg_ratio'] = df[final_target_col] / df['predicted_avg_price']

# Update numerical columns list to include new features
num_cols = num_cols + [
    time_series_target,            # Actual average price by time and location (this would be unknown for new data)
    'predicted_avg_price',         # Predicted average price (model can generate this for new data)
    'month_sin', 'month_cos',      # Cyclical time features
    'month', 'quarter', 'year',    # Time components
    'price_to_avg_ratio'           # Price ratio
]

# Feature engineering
df["log_dientich"] = np.log1p(df["Diện tích (m2)"])

# Remove non-feature columns for model training
X = df.drop(columns=[final_target_col, "Thời điểm hiệu lực", "month_year"])
y = np.log1p(df[final_target_col])

# =================== SPLIT DATA =================== #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Train-test split: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")

# =================== PREPROCESSOR =================== #
numeric_features = num_cols + ["log_dientich"]
categorical_features = [
    "Thành phố/Quận/Huyện/Thị xã", "Xã/Phường/Thị trấn",
    "Vị trí trong khung giá", "Lợi thế kinh doanh"
]
target_encoded_features = ["Đường phố"]

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), categorical_features),
    ("target_enc", TargetEncodingWrapper(), target_encoded_features),
], remainder='drop')  # Drop columns not specified

# =================== PIPELINE FOR LIGHTGBM =================== #
# LightGBM configuration with GPU if available
lgb_params = {
    'random_state': 42,
    'force_col_wise': True,
    'verbose': -1
}

if use_gpu:
    lgb_params.update({
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
    })

lgb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", lgb.LGBMRegressor(**lgb_params))
])

# =================== OPTUNA TUNING FOR LIGHTGBM =================== #
def lgb_objective(trial):
    # Suggest hyperparameters
    param = {
        "regressor__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "regressor__n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "regressor__max_depth": trial.suggest_int("max_depth", 3, 12),
        "regressor__num_leaves": trial.suggest_int("num_leaves", 7, 200),
        "regressor__min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "regressor__subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "regressor__colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "regressor__reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "regressor__reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    
    # Set pipeline parameters
    pipeline_with_params = lgb_pipeline.set_params(**param)
    
    # Cross-validation using KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Fit model
        pipeline_with_params.fit(X_fold_train, y_fold_train)
        
        # Predict and evaluate using our custom combined threshold metric
        y_pred = pipeline_with_params.predict(X_fold_val)
        
        # We want to maximize the percentage of predictions within thresholds
        y_true_exp = np.expm1(y_fold_val)
        y_pred_exp = np.expm1(y_pred)
        
        score = combined_threshold_score(y_true_exp, y_pred_exp)
        scores.append(score)
    
    # Return mean score across folds
    return np.mean(scores)

# Create and run the study
print("\n🔍 Optimizing LightGBM model to maximize predictions within APE thresholds...")
lgb_study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
lgb_study.optimize(lgb_objective, n_trials=50)
best_lgb_params = lgb_study.best_params
print("✅ Best LightGBM parameters for maximizing APE thresholds:", best_lgb_params)
print(f"✅ Best combined APE threshold score: {lgb_study.best_value:.2f}")

# Save best LightGBM parameters
save_model_params("lightgbm", best_lgb_params, results_dir=RESULTS_DIR)

# =================== FINAL TRAINING =================== #
# Convert parameter names to include regressor__ prefix
lgb_pipeline_params = {f"regressor__{k}": v for k, v in best_lgb_params.items()}
lgb_pipeline.set_params(**lgb_pipeline_params)

# Add GPU parameters again if available
if use_gpu:
    lgb_pipeline.set_params(
        regressor__device='gpu',
        regressor__gpu_platform_id=0,
        regressor__gpu_device_id=0
    )

print("\n🔧 Training LightGBM model...")
lgb_pipeline.fit(X_train, y_train)

# =================== EVALUATION =================== #
def evaluate_model(model, X_data, y_data_log, label):
    # Predict on log scale
    y_pred_log = model.predict(X_data)
    
    # Convert from log scale for evaluation
    y_true = np.expm1(y_data_log)
    y_pred = np.expm1(y_pred_log)
    
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape_val = mape_score(y_true, y_pred)
    
    # Calculate percentage of predictions within thresholds
    within_10pct = ape_threshold_score(y_true, y_pred, threshold=0.1)
    within_15pct = ape_threshold_score(y_true, y_pred, threshold=0.15)
    
    print(f"{label} - R²: {r2:.4f}")
    print(f"{label} - RMSE: {rmse:.2f}")
    print(f"{label} - MAPE: {mape_val:.2f}%")
    print(f"{label} - Predictions with APE < 10%: {within_10pct:.2f}%")
    print(f"{label} - Predictions with APE < 15%: {within_15pct:.2f}%")
    
    return {
        "r2": r2,
        "rmse": rmse,
        "mape": mape_val,
        "within_10pct": within_10pct,
        "within_15pct": within_15pct
    }

# Evaluate LightGBM model
print("\n📊 Evaluating LightGBM model...")
lgb_metrics_train = evaluate_model(lgb_pipeline, X_train, y_train, "LightGBM (Train)")
lgb_metrics_test = evaluate_model(lgb_pipeline, X_test, y_test, "LightGBM (Test)")

# =================== FEATURE IMPORTANCE =================== #
print("\n📊 Analyzing feature importance...")
# Get feature importances from the LightGBM model
lgbm_model = lgb_pipeline.named_steps['regressor']
feature_importance = lgbm_model.feature_importances_

# Get feature names from the preprocessor
try:
    # Get column transformer output feature names
    preprocessor_feature_names = preprocessor.get_feature_names_out()
    
    # Create DataFrame with importances
    importance_df = pd.DataFrame({
        'Feature': preprocessor_feature_names,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)
    
    # Print top 20 most important features
    print("\nTop 20 most important features:")
    print(importance_df.head(20))
    
    # Save feature importance to CSV
    importance_df.to_csv(f"{RESULTS_DIR}/feature_importance.csv", index=False)
except Exception as e:
    print(f"Error getting feature names: {e}")
    print("Saving feature importance by index instead")
    
    importance_df = pd.DataFrame({
        'Feature_Index': range(len(feature_importance)),
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)
    
    importance_df.to_csv(f"{RESULTS_DIR}/feature_importance_by_index.csv", index=False)

# =================== SAVE MODEL =================== #
# Save model components
model_dir = os.path.join(RESULTS_DIR, "models")
os.makedirs(model_dir, exist_ok=True)

model_filename = os.path.join(model_dir, "lightgbm_model.pkl")
x_cols_filename = os.path.join(model_dir, "X_train_columns.pkl")
data_info_filename = os.path.join(model_dir, "data_info.json")

# Save LightGBM model
joblib.dump(lgb_pipeline, model_filename)

# Save column information for prediction
joblib.dump(X_train.columns.tolist(), x_cols_filename)

# Save information about data structure
data_info = {
    "location_cols": location_cols,
    "time_series_target": time_series_target,
    "final_target": final_target_col,
    "date_col": date_col,
    "categorical_cols": categorical_features,
    "target_encoded_cols": target_encoded_features,
    "numeric_cols": numeric_features
}

with open(data_info_filename, 'w') as f:
    import json
    json.dump(data_info, f, indent=4)

# Save summary of model performance
training_summary = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "average_price_model": {
        "train_metrics": avg_price_train_metrics,
        "test_metrics": avg_price_test_metrics
    },
    "lightgbm": {
        "params": best_lgb_params,
        "train_metrics": lgb_metrics_train,
        "test_metrics": lgb_metrics_test
    },
    "model_files": {
        "avg_price_model": os.path.join(avg_price_model_dir, "avg_price_model.pkl"),
        "lightgbm_model": model_filename,
        "X_train_columns": x_cols_filename,
        "data_info": data_info_filename
    }
}

# Save the complete training summary
with open(f"{RESULTS_DIR}/training_summary.json", 'w') as f:
    import json
    json.dump(training_summary, f, indent=4)

print(f"\n✅ Average price model saved to {os.path.join(avg_price_model_dir, 'avg_price_model.pkl')}")
print(f"✅ LightGBM model saved to {model_filename}")
print(f"✅ Column information saved to {x_cols_filename}")
print(f"✅ Complete training summary saved to {RESULTS_DIR}/training_summary.json")

print("\n🎉 Training complete!")

if __name__ == "__main__":
    # Code will execute when run directly
    pass