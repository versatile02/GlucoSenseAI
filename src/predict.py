import os
import torch
import numpy as np
import pandas as pd
from src.preprocessing import DiabetesPreprocessor
from src.model import DiabetesANN

class DiabetesPredictor:
    """
    Inference interface for predicting diabetes risk.
    Loads the trained model weights and preprocessor to perform inference.
    """
    def __init__(self, model_path: str = "models/best_model.pth", preprocessor_path: str = "models/preprocessor.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")
            
        # Load Preprocessor
        self.preprocessor = DiabetesPreprocessor.load(preprocessor_path)
        
        # Load Model
        self.model = DiabetesANN(input_dim=8, hidden_dim1=32, hidden_dim2=16)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.feature_names = self.preprocessor.feature_names

    def validate_inputs(self, input_data: dict) -> dict:
        """
        Validates the input parameters. If values are out of medically normal bounds,
        raises ValueError or corrects them with warning messages.
        """
        validated = {}
        for name in self.feature_names:
            if name not in input_data:
                raise ValueError(f"Missing required parameter: {name}")
            try:
                validated[name] = float(input_data[name])
            except ValueError:
                raise ValueError(f"Parameter '{name}' must be a numeric value.")
                
        # Medically plausible range validations (with basic sanity warnings)
        if validated['Age'] < 0 or validated['Age'] > 120:
            raise ValueError("Age must be between 0 and 120 years.")
            
        if validated['Pregnancies'] < 0 or validated['Pregnancies'] > 25:
            raise ValueError("Pregnancies must be a non-negative integer between 0 and 25.")
            
        if validated['BMI'] < 0 or validated['BMI'] > 100:
            raise ValueError("BMI must be between 0 and 100.")
            
        if validated['Glucose'] < 0 or validated['Glucose'] > 500:
            raise ValueError("Glucose must be between 0 and 500 mg/dL.")
            
        if validated['BloodPressure'] < 0 or validated['BloodPressure'] > 300:
            raise ValueError("Blood Pressure must be between 0 and 300 mmHg.")
            
        if validated['SkinThickness'] < 0 or validated['SkinThickness'] > 100:
            raise ValueError("Triceps skin fold thickness must be between 0 and 100 mm.")
            
        if validated['Insulin'] < 0 or validated['Insulin'] > 1500:
            raise ValueError("2-Hour serum insulin must be between 0 and 1500 mu U/ml.")
            
        if validated['DiabetesPedigreeFunction'] < 0 or validated['DiabetesPedigreeFunction'] > 3.0:
            raise ValueError("Diabetes Pedigree Function must be between 0.0 and 3.0.")
            
        return validated

    def predict(self, raw_input: dict) -> dict:
        """
        Accepts raw feature dictionary, processes it, and computes prediction.
        Returns:
            dict containing prediction class, probability, risk level, and description.
        """
        # Validate inputs
        validated_input = self.validate_inputs(raw_input)
        
        # Convert dictionary to DataFrame with correct column ordering
        df_input = pd.DataFrame([validated_input], columns=self.feature_names)
        
        # Preprocess features (imputation + scaling)
        X_scaled = self.preprocessor.transform(df_input)
        
        # Convert to Tensor
        x_tensor = torch.FloatTensor(X_scaled)
        
        # Forward pass
        with torch.no_grad():
            logit = self.model(x_tensor)
            prob = torch.sigmoid(logit).item()
            
        # Determine risk level and prediction class
        is_diabetic = prob >= 0.5
        prediction_class = 1 if is_diabetic else 0
        
        if prob < 0.3:
            risk_level = "Low Risk"
            interpretation = "The clinical parameters suggest a low likelihood of diabetes risk. Maintain a healthy lifestyle."
        elif prob < 0.7:
            risk_level = "Moderate Risk"
            interpretation = "The parameters indicate moderate indicators of diabetes risk. It is recommended to consult a doctor for a professional checkup and monitor diet."
        else:
            risk_level = "High Risk"
            interpretation = "The parameters indicate high risk indicators. It is highly recommended to seek medical advice, run a HbA1c/Fasting blood sugar test, and consult a physician."
            
        return {
            'prediction_class': prediction_class,
            'probability': prob,
            'risk_level': risk_level,
            'interpretation': interpretation,
            'disclaimer': (
                "DISCLAIMER: This system is a machine-learning-based educational risk-prediction model "
                "and does NOT represent a medical diagnosis. Please consult a qualified clinical "
                "healthcare professional for formal medical advice, diagnosis, or treatment."
            )
        }

if __name__ == "__main__":
    # Test prediction
    predictor = DiabetesPredictor()
    sample_input = {
        'Pregnancies': 6,
        'Glucose': 148,
        'BloodPressure': 72,
        'SkinThickness': 35,
        'Insulin': 0,  # Will be imputed
        'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627,
        'Age': 50
    }
    result = predictor.predict(sample_input)
    print("Sample Input:", sample_input)
    print("Prediction Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
