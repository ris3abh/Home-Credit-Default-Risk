import pandas as pd
import numpy as np
import joblib
import logging
import argparse
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ModelTester:
    def __init__(self, model_path):
        """Initialize ModelTester with path to saved model."""
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load the model
        self.logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)
        
        # Create output directory for plots
        self.output_dir = Path('test_results')
        self.output_dir.mkdir(exist_ok=True)

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

    def load_and_preprocess_data(self, test_data_path):
        """Load and preprocess test data."""
        self.logger.info(f"Loading test data from {test_data_path}")
        df_test = self.reduce_memory_usage(df_test)
        X_test = self.preprocess_data(df_test, is_training=False)

        return self.test_data

    def make_predictions(self, X):
        """Make predictions using the loaded model."""
        self.logger.info("Making predictions...")
        self.predictions = self.model.predict(X)
        self.prediction_proba = self.model.predict_proba(X)[:, 1]
        return self.predictions, self.prediction_proba

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.output_dir / 'confusion_matrix.png')
        plt.close()

    def plot_roc_curve(self, y_true, y_pred_proba):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(self.output_dir / 'roc_curve.png')
        plt.close()

    def plot_precision_recall_curve(self, y_true, y_pred_proba):
        """Plot Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.savefig(self.output_dir / 'precision_recall_curve.png')
        plt.close()

    def analyze_feature_importance(self):
        """Analyze and plot feature importance."""
        if hasattr(self.model['classifier'], 'feature_importances_'):
            # Get feature names from the preprocessor
            feature_names = (
                self.model['preprocessor']
                .named_transformers_['num'].get_feature_names_out()
            )
            
            # Get feature importance scores
            importance_scores = self.model['classifier'].feature_importances_
            
            # Create DataFrame of features and their importance scores
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_scores
            })
            feature_importance = feature_importance.sort_values(
                'importance', ascending=False
            )
            
            # Plot feature importance
            plt.figure(figsize=(12, 6))
            sns.barplot(data=feature_importance.head(20), 
                       x='importance', y='feature')
            plt.title('Top 20 Most Important Features')
            plt.xlabel('Feature Importance Score')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'feature_importance.png')
            plt.close()
            
            return feature_importance

    def generate_prediction_report(self, y_true, y_pred, y_pred_proba):
        """Generate a comprehensive prediction report."""
        report = classification_report(y_true, y_pred)
        
        # Save report to file
        with open(self.output_dir / 'prediction_report.txt', 'w') as f:
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n\nPrediction Statistics:\n")
            f.write(f"Total Samples: {len(y_true)}\n")
            f.write(f"Positive Predictions: {sum(y_pred)}\n")
            f.write(f"Negative Predictions: {len(y_pred) - sum(y_pred)}\n")
            
        return report

    def run_complete_analysis(self, test_data_path):
        """Run complete model testing analysis."""
        # Load and process test data
        test_data = self.load_and_preprocess_data(test_data_path)
        
        # Get predictions
        X_test = test_data.drop('TARGET', axis=1)
        y_test = test_data['TARGET']
        y_pred, y_pred_proba = self.make_predictions(X_test)
        
        # Generate visualizations
        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_roc_curve(y_test, y_pred_proba)
        self.plot_precision_recall_curve(y_test, y_pred_proba)
        
        # Analyze feature importance
        feature_importance = self.analyze_feature_importance()
        
        # Generate report
        report = self.generate_prediction_report(y_test, y_pred, y_pred_proba)
        
        self.logger.info(f"\nClassification Report:\n{report}")
        self.logger.info(f"Results saved in {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Test Credit Fraud Detection Model')
    parser.add_argument(
        '--model-path',
        default='../models/credit_fraud_model.pkl',
        help='Path to trained model'
    )
    parser.add_argument(
        '--test-data',
        default='../data/test_data.csv',
        help='Path to test data'
    )
    
    args = parser.parse_args()
    
    # Initialize and run tester
    tester = ModelTester(args.model_path)
    tester.run_complete_analysis(args.test_data)

if __name__ == "__main__":
    main()


# initially we do not have TARGET Label in the test data, so we will care about the testing later.