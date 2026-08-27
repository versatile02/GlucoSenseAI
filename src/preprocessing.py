import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DiabetesPreprocessor:
    """
    Handles preprocessing for the diabetes risk dataset.
    Replaces medically implausible zero values with training set medians
    and standardizes the features.
    """
    def __init__(self):
        # Columns where 0 indicates missing/implausible value
        self.zero_impute_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        self.medians_ = {}
        self.scaler_ = StandardScaler()
        self.feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]

    def fit(self, X: pd.DataFrame, y=None):
        """
        Learns the medians and scaling parameters from the training set.
        """
        # Ensure we work with a DataFrame copy
        X_df = pd.DataFrame(X, columns=self.feature_names).copy()
        
        # Calculate medians for the target columns where value is greater than 0
        for col in self.zero_impute_cols:
            non_zero_values = X_df[col][X_df[col] > 0]
            # Fallback to 0 if all values are 0 (unlikely)
            self.medians_[col] = non_zero_values.median() if len(non_zero_values) > 0 else 0.0

        # Create a temporary copy to fit the scaler
        X_imputed = X_df.copy()
        for col in self.zero_impute_cols:
            X_imputed[col] = X_imputed[col].replace(0, self.medians_[col])
            
        self.scaler_.fit(X_imputed)
        return self

    def transform(self, X) -> np.ndarray:
        """
        Imputes and standardizes features based on learned parameters.
        """
        # If input is a numpy array, convert it to DataFrame for column alignment
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names).copy()
        else:
            X_df = pd.DataFrame(X).copy()
            
        # Impute zeros with learned medians
        for col in self.zero_impute_cols:
            X_df[col] = X_df[col].replace(0, self.medians_[col])
            
        # Scale all features
        X_scaled = self.scaler_.transform(X_df)
        return X_scaled

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def save(self, filepath: str):
        """Saves the preprocessor instance to a file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> 'DiabetesPreprocessor':
        """Loads a preprocessor instance from a file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def load_raw_data(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads raw CSV data and separates it into features (X) and target (y).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Simple validation
    required_cols = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the dataset.")
            
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    return X, y
