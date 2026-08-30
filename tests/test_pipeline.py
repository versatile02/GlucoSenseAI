import pytest
import numpy as np
import pandas as pd
import torch
from src.preprocessing import DiabetesPreprocessor
from src.model import DiabetesANN
from src.predict import DiabetesPredictor

def test_preprocessor_imputation():
    """Verify that medically implausible zero values are correctly imputed."""
    # Create sample raw data with zeros in implausible columns
    raw_data = pd.DataFrame({
        'Pregnancies': [1, 2],
        'Glucose': [0, 120],       # Implausible zero
        'BloodPressure': [80, 0],   # Implausible zero
        'SkinThickness': [20, 15],
        'Insulin': [0, 85],         # Implausible zero
        'BMI': [25.0, 30.0],
        'DiabetesPedigreeFunction': [0.5, 0.6],
        'Age': [30, 45]
    })
    
    preprocessor = DiabetesPreprocessor()
    # Fit and transform
    scaled_data = preprocessor.fit_transform(raw_data)
    
    # Check that medians were calculated and are greater than 0
    assert preprocessor.medians_['Glucose'] == 120.0
    assert preprocessor.medians_['BloodPressure'] == 80.0
    assert preprocessor.medians_['Insulin'] == 85.0
    
    # Check transformation on a new sample
    test_sample = pd.DataFrame([{
        'Pregnancies': 0, 'Glucose': 0, 'BloodPressure': 0, 'SkinThickness': 0,
        'Insulin': 0, 'BMI': 0, 'DiabetesPedigreeFunction': 0.1, 'Age': 25
    }])
    
    # Transform test_sample
    transformed_sample = preprocessor.transform(test_sample)
    
    # The preprocessor contains the scaler inside. Let's inverse-transform the values to check the imputed values
    imputed_values = preprocessor.scaler_.inverse_transform(transformed_sample)[0]
    
    # Check that imputed values correspond to the fitted training medians
    # Indices: Pregnancies (0), Glucose (1), BP (2), Skin (3), Insulin (4), BMI (5), DPF (6), Age (7)
    assert imputed_values[1] == 120.0  # Glucose imputed
    assert imputed_values[2] == 80.0   # BP imputed
    assert imputed_values[4] == 85.0   # Insulin imputed

def test_model_forward_shape():
    """Verify that the neural network accepts input (B, 8) and outputs (B, 1)."""
    model = DiabetesANN(input_dim=8, hidden_dim1=16, hidden_dim2=8)
    batch_size = 4
    dummy_input = torch.randn(batch_size, 8)
    output = model(dummy_input)
    
    assert output.shape == (batch_size, 1)

def test_predictor_flow():
    """Verify inference pipeline and validation constraints."""
    predictor = DiabetesPredictor()
    
    # Valid input prediction test
    valid_input = {
        'Pregnancies': 2,
        'Glucose': 110,
        'BloodPressure': 75,
        'SkinThickness': 20,
        'Insulin': 100,
        'BMI': 24.5,
        'DiabetesPedigreeFunction': 0.35,
        'Age': 28
    }
    
    res = predictor.predict(valid_input)
    
    assert 'prediction_class' in res
    assert 'probability' in res
    assert 'risk_level' in res
    assert 'interpretation' in res
    assert 'disclaimer' in res
    assert 0.0 <= res['probability'] <= 1.0
    
    # Invalid inputs tests (should raise ValueErrors)
    invalid_age = valid_input.copy()
    invalid_age['Age'] = 150  # Over 120
    with pytest.raises(ValueError, match="Age must be between 0 and 120 years."):
        predictor.predict(invalid_age)
        
    invalid_bmi = valid_input.copy()
    invalid_bmi['BMI'] = -5  # Negative BMI
    with pytest.raises(ValueError, match="BMI must be between 0 and 100."):
        predictor.predict(invalid_bmi)

    invalid_type = valid_input.copy()
    invalid_type['Glucose'] = "not_a_number"
    with pytest.raises(ValueError, match="Parameter 'Glucose' must be a numeric value."):
        predictor.predict(invalid_type)
