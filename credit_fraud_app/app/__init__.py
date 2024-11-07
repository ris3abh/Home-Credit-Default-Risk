from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import logging
import os

# Get absolute paths
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))

# Initialize Flask app with explicit template and static paths
app = Flask(__name__,
           template_folder=template_dir,
           static_folder=static_dir)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Debug information
logger.debug(f"Template directory: {template_dir}")
logger.debug(f"Static directory: {static_dir}")
logger.debug(f"Current working directory: {os.getcwd()}")
logger.debug(f"Directory contents: {os.listdir(os.getcwd())}")

if os.path.exists(template_dir):
    logger.debug(f"Template directory contents: {os.listdir(template_dir)}")
else:
    logger.debug(f"Template directory not found!")

# Model path from environment variable or default
MODEL_PATH = os.path.join(os.getcwd(), 'app', 'models', 'model.pkl')

# Load the model
try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}!")
except Exception as e:
    logger.error(f"Error loading model from {MODEL_PATH}: {str(e)}")
    model = None

def prepare_input_data(form_data):
    """Convert form data to model input format with feature engineering"""
    try:
        # First create the basic dataframe
        data = {
            'CODE_GENDER': [form_data.get('gender')],
            'NAME_EDUCATION_TYPE': [form_data.get('education')],
            'FLAG_OWN_CAR': [form_data.get('own_car')],
            'FLAG_OWN_REALTY': [form_data.get('own_realty')],
            'NAME_INCOME_TYPE': [form_data.get('income_type')],
            'REG_CITY_NOT_WORK_CITY': [int(form_data.get('diff_city', 0))],
            'CNT_CHILDREN': [int(form_data.get('children', 0))],
            'AMT_INCOME_TOTAL': [float(form_data.get('income', 0))],
            'AMT_CREDIT': [float(form_data.get('credit_amount', 0))],
            'AMT_ANNUITY': [float(form_data.get('annuity', 0))],
            'AMT_GOODS_PRICE': [float(form_data.get('goods_price', 0))],
            'DAYS_BIRTH': [int(form_data.get('age', 0)) * -365],
            'DAYS_EMPLOYED': [float(form_data.get('years_employed', 0)) * -365],
            'DAYS_REGISTRATION': [float(form_data.get('years_registration', 0)) * -365],
            'EXT_SOURCE_1': [float(form_data.get('ext_source_1', 0))],
            'EXT_SOURCE_2': [float(form_data.get('ext_source_2', 0))],
            'EXT_SOURCE_3': [float(form_data.get('ext_source_3', 0))]
        }
        
        df = pd.DataFrame(data)
        
        # Calculate RATE_OF_LOAN
        df['RATE_OF_LOAN'] = (df['AMT_ANNUITY'] / df['AMT_CREDIT']).round(6)
        
        # Log transform monetary values
        for col in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']:
            df[col] = np.log(df[col] + 1)
        
        # Convert days to years
        df["AGE_YEARS"] = df["DAYS_BIRTH"]/-365
        df["YEARS_EMPLOYED"] = df["DAYS_EMPLOYED"]/-365
        df["YEARS_REGISTRATION"] = df["DAYS_REGISTRATION"] / -365
        
        logger.info("Input features created successfully")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Data shape: {df.shape}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error in prepare_input_data: {str(e)}")
        raise

@app.route('/')
def home():
    try:
        logger.debug(f"Template folder: {app.template_folder}")
        logger.debug(f"Static folder: {app.static_folder}")
        logger.debug(f"Current directory: {os.getcwd()}")
        logger.debug(f"Directory contents: {os.listdir('.')}")
        
        template_path = os.path.join(app.template_folder or '', 'index.html')
        logger.debug(f"Looking for template at: {template_path}")
        
        if model is None:
            logger.error("Model not loaded properly")
            return "Error: Model not loaded properly", 500
            
        return render_template('index.html')
    except Exception as e:
        logger.exception(f"Error in home route:")
        return f"Error: {str(e)}", 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded properly'}), 500
        
        # Log the incoming form data
        logger.info("Received form data")
        logger.debug(f"Form data: {request.form}")
        
        # Get form data and prepare input
        input_data = prepare_input_data(request.form)
        
        # Log the processed input data
        logger.debug("Processed input data:")
        logger.debug(input_data.to_dict())
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        result = {
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': 'High Risk' if probability > 0.5 else 'Low Risk',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Log the prediction result
        logger.info(f"Prediction result: {result}")
        
        return render_template('result.html', result=result)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400

# Health check endpoint
@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

def create_app():
    """Factory function for creating the Flask app"""
    return app

application = app

if __name__ == '__main__':
    # Only use debug mode when running directly (not with gunicorn)
    app.run(debug=True, host='0.0.0.0', port=5000)
else:
    # When running with gunicorn, use production settings
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)