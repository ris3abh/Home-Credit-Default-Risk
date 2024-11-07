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
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import yaml
import os
import json

class ModelEnvironment(Enum):
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"

@dataclass
class ModelVersion:
    major: int
    minor: int
    patch: int
    environment: ModelEnvironment
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def __str__(self):
        base = f"v{self.major}.{self.minor}.{self.patch}"
        env = f"-{self.environment.value}"
        date = f"-{self.timestamp.strftime('%Y%m%d')}"
        return f"{base}{env}{date}"
    
    @classmethod
    def from_string(cls, version_string: str):
        # Parse version string like "v1.2.3-prod-20240307"
        version_parts = version_string.split('-')
        version_numbers = version_parts[0][1:].split('.')
        return cls(
            major=int(version_numbers[0]),
            minor=int(version_numbers[1]),
            patch=int(version_numbers[2]),
            environment=ModelEnvironment(version_parts[1]),
            timestamp=datetime.strptime(version_parts[2], '%Y%m%d')
        )

class ModelRegistry:
    def __init__(self, registry_path="model_registry"):
        self.registry_path = registry_path
        self.logger = logging.getLogger(__name__)
        os.makedirs(registry_path, exist_ok=True)
        
    def _get_version_path(self, version: ModelVersion):
        return os.path.join(self.registry_path, str(version))
    
    def save_model_version(self, model, metadata: dict, version: ModelVersion):
        version_path = self._get_version_path(version)
        os.makedirs(version_path, exist_ok=True)
        
        # Save model
        model_path = os.path.join(version_path, "model.pkl")
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata.update({
            "version": str(version),
            "saved_at": datetime.now().isoformat()
        })
        metadata_path = os.path.join(version_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved model version {version} to {version_path}")
    
    def load_model_version(self, version: ModelVersion):
        version_path = self._get_version_path(version)
        model_path = os.path.join(version_path, "model.pkl")
        metadata_path = os.path.join(version_path, "metadata.json")
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model version {version} not found")
            
        model = joblib.load(model_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        return model, metadata

class CreditFraudModel:
    def __init__(self, version: ModelVersion = None):
        # Initialize version
        self.version = version or ModelVersion(0, 1, 0, ModelEnvironment.DEV)
        self.registry = ModelRegistry()
        
        # Original initialization code
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
        
        # Initialize model changelog
        self.changelog = []

    def log_change(self, change_type: str, description: str):
        """Log a change to the model."""
        self.changelog.append({
            'version': str(self.version),
            'type': change_type,
            'description': description,
            'timestamp': datetime.now().isoformat()
        })

    def save_model(self, path=None, metadata=None):
        """Save the model with version information."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Prepare metadata
        metadata = metadata or {}
        metadata.update({
            'feature_columns': {
                'categorical': self.categorical_columns,
                'numerical': self.numerical_columns
            },
            'changelog': self.changelog,
            'model_parameters': self.model.named_steps['classifier'].get_params(),
            'training_date': datetime.now().isoformat()
        })
        
        # Save to registry
        self.registry.save_model_version(self.model, metadata, self.version)
        
        # If additional path specified, save there too
        if path:
            joblib.dump(self.model, path)
            self.logger.info(f"Model also saved to {path}")

    def load_model(self, version: ModelVersion = None):
        """Load a specific version of the model."""
        version = version or self.version
        self.model, metadata = self.registry.load_model_version(version)
        self.version = version
        self.changelog = metadata.get('changelog', [])
        return self.model

    # [Previous methods remain the same: reduce_memory_usage, preprocess_data, setup_pipeline, etc.]

    def train_and_evaluate(self, train_path, test_path=None, increment_version=True):
        """Train the model and optionally increment version."""
        # Original training code
        df_train = pd.read_csv(train_path)
        df_train = self.reduce_memory_usage(df_train)
        df_train = self.preprocess_data(df_train, is_training=True)
        
        X_train = df_train.drop(columns=["TARGET"])
        y_train = df_train["TARGET"]
        
        negative_samples = (y_train == 0).sum()
        positive_samples = (y_train == 1).sum()
        scale_pos_weight = negative_samples / positive_samples
        
        self.setup_pipeline(scale_pos_weight)
        self.model.fit(X_train, y_train)
        
        # Log the training
        self.log_change(
            'training',
            f'Model trained on {train_path} with {len(X_train)} samples'
        )
        
        # Handle test data if provided
        if test_path:
            df_test = pd.read_csv(test_path)
            df_test = self.reduce_memory_usage(df_test)
            X_test = self.preprocess_data(df_test, is_training=False)
            
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            predictions_df = pd.DataFrame({
                'prediction': y_pred,
                'probability': y_pred_proba
            })
            predictions_df.to_csv('predictions.csv', index=False)
            
            # Log the evaluation
            self.log_change(
                'evaluation',
                f'Model evaluated on {test_path} with {len(X_test)} samples'
            )
        
        # Increment version if requested
        if increment_version:
            self.version = ModelVersion(
                major=self.version.major,
                minor=self.version.minor + 1,
                patch=0,
                environment=self.version.environment
            )
        
        return self.model

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
    parser.add_argument('--environment',
                       choices=['dev', 'staging', 'prod'],
                       default='dev',
                       help='Model environment')
    args = parser.parse_args()
    
    # Initialize model with version
    initial_version = ModelVersion(
        major=1,
        minor=0,
        patch=0,
        environment=ModelEnvironment(args.environment)
    )
    
    model = CreditFraudModel(version=initial_version)
    trained_model = model.train_and_evaluate(
        train_path=args.train_path,
        test_path=args.test_path
    )
    model.save_model(path=args.model_output)

if __name__ == "__main__":
    main()