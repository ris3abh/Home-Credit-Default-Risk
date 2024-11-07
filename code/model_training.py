import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, precision_score, 
                           recall_score, roc_auc_score, classification_report)
from xgboost import XGBClassifier
import joblib
import logging
import argparse

class CreditFraudModel:
    def __init__(self):
        # Initialize class attributes
        self.model = None
        self.preprocessor = None
        self.selected_columns = [
            'CODE_GENDER', 'NAME_EDUCATION_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
            'NAME_INCOME_TYPE', 'REG_CITY_NOT_WORK_CITY', 'CNT_CHILDREN',
            'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
            'AGE_YEARS', 'YEARS_EMPLOYED', 'YEARS_REGISTRATION', 'EXT_SOURCE_1',
            'EXT_SOURCE_2', 'EXT_SOURCE_3', 'RATE_OF_LOAN', 'TARGET'
        ]
        self.categorical_columns = [
            'CODE_GENDER', 'CNT_CHILDREN', 'NAME_EDUCATION_TYPE', 'FLAG_OWN_CAR',
            'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 'REG_CITY_NOT_WORK_CITY'
        ]
        self.numerical_columns = [
            'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
            'AGE_YEARS', 'YEARS_EMPLOYED', 'YEARS_REGISTRATION', 'EXT_SOURCE_1',
            'EXT_SOURCE_2', 'EXT_SOURCE_3', 'RATE_OF_LOAN'
        ]
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def reduce_memory_usage(self, df):
        """Reduce memory usage of the dataframe."""
        start_mem = df.memory_usage().sum() / 1024**2
        self.logger.info(f'Memory usage of dataframe is {start_mem:.2f} MB')
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        
        end_mem = df.memory_usage().sum() / 1024**2
        self.logger.info(f'Memory usage after optimization is: {end_mem:.2f} MB')
        self.logger.info(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
        
        return df

    def preprocess_data(self, df, is_training=True):
        """Preprocess the data with feature engineering."""
        df = df.copy()
        
        # Calculate rate of loan
        df['RATE_OF_LOAN'] = (df['AMT_ANNUITY'] / df['AMT_CREDIT']).round(6)
        
        # Log transform monetary values
        for col in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']:
            df[col] = np.log(df[col] + 1)
        
        # Convert days to years
        df["AGE_YEARS"] = df["DAYS_BIRTH"]/-365
        df["YEARS_EMPLOYED"] = df["DAYS_EMPLOYED"]/-365
        df["YEARS_REGISTRATION"] = df["DAYS_REGISTRATION"] / -365
        
        # If training data, return all columns including TARGET
        # If test data, return all columns except TARGET
        columns_to_return = self.selected_columns if is_training else [col for col in self.selected_columns if col != 'TARGET']
        return df[columns_to_return]


    def setup_pipeline(self, scale_pos_weight):
        """Set up the preprocessing and model pipeline."""
        numerical_transformer = SimpleImputer(strategy='mean')
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_columns),
                ('cat', categorical_transformer, self.categorical_columns)
            ])

        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                min_child_weight=10,
                max_depth=3,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight
            ))
        ])

    def train_and_evaluate(self, train_path, test_path=None):
        """Train the model on training data and evaluate if test data is provided."""
        # Load and preprocess training data
        self.logger.info("Loading and preprocessing training data...")
        df_train = pd.read_csv(train_path)
        df_train = self.reduce_memory_usage(df_train)
        df_train = self.preprocess_data(df_train, is_training=True)
        
        # Split features and target for training data
        X_train = df_train.drop(columns=["TARGET"])
        y_train = df_train["TARGET"]
        
        # Calculate class weights from training data
        negative_samples = (y_train == 0).sum()
        positive_samples = (y_train == 1).sum()
        scale_pos_weight = negative_samples / positive_samples
        
        # Setup pipeline
        self.setup_pipeline(scale_pos_weight)
        
        # Train model
        self.logger.info("Training model...")
        self.model.fit(X_train, y_train)
        
        # If test path is provided, evaluate on test data
        if test_path:
            self.logger.info("Loading and preprocessing test data...")
            df_test = pd.read_csv(test_path)
            df_test = self.reduce_memory_usage(df_test)
            X_test = self.preprocess_data(df_test, is_training=False)
            
            # Make predictions on test data
            self.logger.info("Making predictions on test data...")
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            # Save predictions
            predictions_df = pd.DataFrame({
                'prediction': y_pred,
                'probability': y_pred_proba
            })
            predictions_df.to_csv('predictions.csv', index=False)
            self.logger.info("Predictions saved to predictions.csv")
        
        return self.model

    def predict(self, data):
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_and_evaluate first.")
        
        # Preprocess the data
        processed_data = self.preprocess_data(data, is_training=False)
        
        # Make predictions
        predictions = self.model.predict(processed_data)
        probabilities = self.model.predict_proba(processed_data)[:, 1]
        return predictions, probabilities


    def save_model(self, path='../models/credit_fraud_model.pkl'):
        """Save the trained model to a file."""
        if self.model is not None:
            joblib.dump(self.model, path)
            self.logger.info(f"Model saved to {path}")
        else:
            self.logger.error("No model to save. Train the model first.")

def main():
    parser = argparse.ArgumentParser(description='Train Credit Fraud Detection Model')
    parser.add_argument('--train-path', 
                       default='../data/application_train.csv',
                       help='Path to training data')
    parser.add_argument('--test-path', 
                       default=None,
                       help='Path to test data (optional)')
    parser.add_argument('--model-output', 
                       default='models/credit_fraud_model.pkl',
                       help='Path to save trained model')
    args = parser.parse_args()
    model = CreditFraudModel()
    trained_model = model.train_and_evaluate(
        train_path=args.train_path,
        test_path=args.test_path
    )
    model.save_model(path=args.model_output)

if __name__ == "__main__":
    main()

# setuptools==65.5.0
# six==1.16.0
# threadpoolctl==3.5.0
# tzdata==2024.1
# matplotlib==3.9.0
# seaborn==0.13.2
# xgboost==2.1.0
# imblearn==0.0
# lightgbm==4.4.0
# joblib==1.4.2
# numpy>=1.24.3,<2.0.0
# pandas==2.0.2
# python-dateutil==2.9.0.post0
# pytz==2024.1
# scikit-learn==1.3.2
# scipy==1.10.1
