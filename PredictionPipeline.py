import pandas as pd
import numpy as np
import os
import joblib
import json
from datetime import datetime
import warnings

from DartPricePredictor import DartPricePredictor
from utils.time_series_utils import create_time_features, evaluate_time_series_model

warnings.filterwarnings("ignore")

class PredictionPipeline:
    """
    Combined prediction pipeline that:
    1. Uses Dart to predict average prices by location and time
    2. Feeds those predictions into LightGBM to predict final prices
    """
    
    def __init__(self, model_dir):
        """
        Initialize prediction pipeline
        
        Args:
            model_dir: Directory containing saved models
        """
        self.model_dir = model_dir
        self.dart_model_dir = os.path.join(model_dir, "dart_models")
        self.lgb_model_path = os.path.join(model_dir, "models/lightgbm_model.pkl")
        self.data_info_path = os.path.join(model_dir, "models/data_info.json")
        self.x_cols_path = os.path.join(model_dir, "models/X_train_columns.pkl")
        
        # Load components
        self._load_components()
        
    def _load_components(self):
        """Load all model components"""
        print("\n🔍 Loading model components...")
        
        # Load LightGBM model
        try:
            self.lgb_model = joblib.load(self.lgb_model_path)
            print(f"✅ Loaded LightGBM model from {self.lgb_model_path}")
        except Exception as e:
            print(f"❌ Error loading LightGBM model: {e}")
            raise
            
        # Load data info
        try:
            with open(self.data_info_path, 'r') as f:
                self.data_info = json.load(f)
            print(f"✅ Loaded data info from {self.data_info_path}")
        except Exception as e:
            print(f"❌ Error loading data info: {e}")
            raise
            
        # Load column information
        try:
            self.x_cols = joblib.load(self.x_cols_path)
            print(f"✅ Loaded column information from {self.x_cols_path}")
        except Exception as e:
            print(f"❌ Error loading column information: {e}")
            raise
            
        # Initialize Dart predictor
        try:
            self.dart_predictor = DartPricePredictor(results_dir=self.dart_model_dir)
            dart_model_path = os.path.join(self.dart_model_dir, "dart_models.pkl")
            self.dart_predictor.load_model(dart_model_path)
            print(f"✅ Loaded Dart models from {dart_model_path}")
        except Exception as e:
            print(f"❌ Error loading Dart models: {e}")
            raise
    
    def preprocess_data(self, df, date_col='Thời điểm hiệu lực'):
        """
        Preprocess input data for prediction
        
        Args:
            df: DataFrame with test data
            date_col: Column containing dates
            
        Returns:
            Preprocessed DataFrame
        """
        print("\n🔄 Preprocessing test data...")
        
        # Create time features
        df = create_time_features(df.copy(), date_col)
        
        # Create additional time features (like in training)
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        # Feature engineering
        df["log_dientich"] = np.log1p(df["Diện tích (m2)"])
        
        return df
    
    def predict(self, df):
        """
        Generate predictions for test data
        
        Args:
            df: DataFrame with test data
            
        Returns:
            DataFrame with predictions
        """
        print("\n🔮 Generating predictions...")
        
        # Get key information from data_info
        location_cols = self.data_info["location_cols"]
        time_series_target = self.data_info["time_series_target"]
        final_target = self.data_info["final_target"]
        date_col = self.data_info["date_col"]
        
        # 1. Preprocess the data
        df_processed = self.preprocess_data(df, date_col)
        
        # 2. Generate average price predictions using Dart
        print("\n🔮 Step 1: Predicting average prices by location and time...")
        df_with_avg = self.dart_predictor.predict(df_processed, location_cols, date_col)
        
        # 3. Rename the Dart predictions to match the expected column name
        df_with_avg[time_series_target] = df_with_avg['predicted_avg_price']
        
        # 4. Create price ratio feature if we have actual prices
        if final_target in df_with_avg.columns:
            df_with_avg['price_to_avg_ratio'] = df_with_avg[final_target] / df_with_avg[time_series_target]
        else:
            # During testing we don't have actual prices, use a default value
            df_with_avg['price_to_avg_ratio'] = 1.0
        
        # 5. Ensure all required columns are present
        for col in self.x_cols:
            if col not in df_with_avg.columns:
                print(f"⚠️ Missing column: {col}, adding with default values")
                if col.startswith('month_') or col.startswith('year') or col.startswith('quarter'):
                    # Time features might have default numeric values
                    df_with_avg[col] = 0
                else:
                    # Other features, use empty string for categorical
                    df_with_avg[col] = ""
        
        # 6. Select only the columns needed for LightGBM (X_train columns)
        X_test = df_with_avg[self.x_cols]
        
        # 7. Generate final predictions using LightGBM
        print("\n🔮 Step 2: Predicting final prices using LightGBM...")
        # LightGBM predicts log-transformed prices
        y_pred_log = self.lgb_model.predict(X_test)
        
        # Convert back from log scale
        y_pred = np.expm1(y_pred_log)
        
        # Add predictions to the original dataframe
        df['predicted_price'] = y_pred
        
        # 8. Evaluate if we have actual prices
        if final_target in df.columns:
            print("\n📊 Evaluating predictions...")
            y_true = df[final_target]
            metrics = evaluate_time_series_model(y_true, y_pred, "Combined Model")
            df['APE'] = np.abs((y_true - y_pred) / (y_true + 1e-10)) * 100
            
            # Add evaluation to result
            result = {
                'predictions': df,
                'metrics': metrics,
                'has_actual_prices': True
            }
        else:
            # No actual prices, just return predictions
            result = {
                'predictions': df,
                'has_actual_prices': False
            }
        
        return result
    
    def save_predictions(self, result, output_dir=None):
        """
        Save prediction results
        
        Args:
            result: Prediction result from predict()
            output_dir: Directory to save results (default: current dir)
            
        Returns:
            Path to saved file
        """
        if output_dir is None:
            output_dir = "."
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save predictions
        predictions_file = os.path.join(output_dir, f"predictions_{timestamp}.xlsx")
        result['predictions'].to_excel(predictions_file, index=False)
        
        # Save metrics if available
        if result.get('has_actual_prices', False):
            metrics_file = os.path.join(output_dir, f"metrics_{timestamp}.json")
            with open(metrics_file, 'w') as f:
                json.dump(result['metrics'], f, indent=4)
            
            print(f"\n✅ Saved predictions to {predictions_file}")
            print(f"✅ Saved metrics to {metrics_file}")
            return predictions_file, metrics_file
        else:
            print(f"\n✅ Saved predictions to {predictions_file}")
            return predictions_file

def main():
    """Main prediction pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run prediction pipeline on test data')
    parser.add_argument('--model_dir', type=str, required=True, 
                        help='Directory containing trained models')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Path to test data Excel file')
    parser.add_argument('--output_dir', type=str, default='prediction_results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    print(f"\n🔧 Initializing prediction pipeline with models from {args.model_dir}")
    pipeline = PredictionPipeline(args.model_dir)
    
    # Load test data
    print(f"\n📂 Loading test data from {args.test_file}")
    test_data = pd.read_excel(args.test_file)
    print(f"✅ Loaded test data with {len(test_data)} rows")
    
    # Generate predictions
    result = pipeline.predict(test_data)
    
    # Save results
    pipeline.save_predictions(result, args.output_dir)
    
    print("\n🎉 Prediction pipeline completed!")

if __name__ == "__main__":
    main()