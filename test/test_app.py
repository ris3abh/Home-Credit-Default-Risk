import pytest
from app import app
import numpy as np
import pandas as pd

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    rv = client.get('/')
    assert rv.status_code == 200
    assert b'Credit Fraud Risk Assessment' in rv.data

def test_prediction_endpoint(client):
    test_data = {
        'gender': 'M',
        'education': 'Higher education',
        'children': '0',
        'age': '30',
        'own_car': 'Y',
        'own_realty': 'Y',
        'income_type': 'Working',
        'years_employed': '5',
        'income': '50000',
        'credit_amount': '100000',
        'annuity': '12000',
        'goods_price': '90000',
        'diff_city': '0',
        'years_registration': '5',
        'ext_source_1': '0.5',
        'ext_source_2': '0.5',
        'ext_source_3': '0.5'
    }
    
    rv = client.post('/predict', data=test_data)
    assert rv.status_code == 200
    assert b'Risk Level' in rv.data

def test_model_preprocessing():
    from app import prepare_input_data
    
    test_form_data = {
        'gender': 'M',
        'education': 'Higher education',
        'children': '0',
        'age': '30',
        'own_car': 'Y',
        'own_realty': 'Y',
        'income_type': 'Working',
        'years_employed': '5',
        'income': '50000',
        'credit_amount': '100000',
        'annuity': '12000',
        'goods_price': '90000',
        'diff_city': '0',
        'years_registration': '5',
        'ext_source_1': '0.5',
        'ext_source_2': '0.5',
        'ext_source_3': '0.5'
    }
    
    df = prepare_input_data(test_form_data)
    assert isinstance(df, pd.DataFrame)
    assert 'RATE_OF_LOAN' in df.columns
    assert 'AGE_YEARS' in df.columns