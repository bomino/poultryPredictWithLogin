from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error

from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

class PoultryWeightPredictor(BaseEstimator, RegressorMixin):
    """Polynomial Regression model for poultry weight prediction."""
    
    def __init__(self, degree=2, fit_intercept=True, include_bias=True):
        """
        Initialize the Polynomial Regression model.
        
        Args:
            degree: Degree of polynomial features
            fit_intercept: Whether to fit intercept in linear regression
            include_bias: Whether to include bias term in polynomial features
        """
        self.degree = int(degree)
        self.fit_intercept = fit_intercept
        self.include_bias = include_bias
        
        # Initialize model pipeline
        self.model = Pipeline([
            ('poly', PolynomialFeatures(
                degree=self.degree,
                include_bias=self.include_bias
            )),
            ('regressor', LinearRegression(
                fit_intercept=self.fit_intercept
            ))
        ])
        
        # Initialize state variables
        self._is_trained = False
        self.feature_names_ = None
        self._feature_importances = None
        self.training_metadata = {}
    
    def _validate_input_data(self, X: np.ndarray, y: Optional[np.ndarray] = None, is_training: bool = True):
        """Validate input data."""
        if X is None:
            raise ValueError("Input features cannot be None")
        if len(X) == 0:
            raise ValueError("Input features cannot be empty")
        if is_training:
            if y is None:
                raise ValueError("Target values cannot be None for training")
            if len(X) != len(y):
                raise ValueError("Number of samples in features and target must match")

    def fit(self, X, y):
        """Fit method for scikit-learn compatibility."""
        return self.train(X, y)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
            feature_names: Optional[List[str]] = None) -> 'PoultryWeightPredictor':
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            feature_names: Optional list of feature names
        """
        try:
            # Validate input data
            self._validate_input_data(X_train, y_train, is_training=True)
            
            # Convert inputs to numpy arrays
            X_train = np.asarray(X_train)
            y_train = np.asarray(y_train)
            
            # Store feature names if provided or create default ones
            if feature_names is not None:
                if len(feature_names) != X_train.shape[1]:
                    raise ValueError(f"Number of feature names ({len(feature_names)}) does not match number of features ({X_train.shape[1]})")
                self.feature_names_ = feature_names
            else:
                self.feature_names_ = [f'feature_{i}' for i in range(X_train.shape[1])]
            
            # Train model
            print(f"Training Polynomial Regression model with degree {self.degree}")
            self.model.fit(X_train, y_train)
            
            # Get polynomial features and coefficients
            poly = self.model.named_steps['poly']
            regressor = self.model.named_steps['regressor']
            
            # Store feature importance information
            self._feature_importances = np.abs(regressor.coef_)
            
            # Mark as trained
            self._is_trained = True
            print("Model training completed successfully, _is_trained set to True")

            # Store training metadata
            self.training_metadata = {
                'n_samples': len(X_train),
                'n_features': X_train.shape[1],
                'n_polynomial_features': len(self._feature_importances),
                'training_date': datetime.now().isoformat(),
                'parameters': self.get_params(),
                'feature_names': self.feature_names_
            }
            
            return self
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self._is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        self._validate_input_data(X, is_training=False)
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        """Evaluate model performance."""
        if not self._is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        try:
            y_pred = self.predict(X_test)
            
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': root_mean_squared_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            }
            
            return metrics, y_pred
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            raise
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self._is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        try:
            poly = self.model.named_steps['poly']
            
            if feature_names is None:
                feature_names = self.feature_names_
            
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(self._feature_importances))]
            
            # Get polynomial feature names
            poly_features = poly.get_feature_names_out(feature_names)
            
            # Create importance dictionary
            importance_dict = dict(zip(poly_features, np.abs(self._feature_importances)))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            print(f"Error getting feature importance: {str(e)}")
            raise
    
    def save(self, filepath: str):
        """Save the trained model."""
        if not self._is_trained:
            raise ValueError("Model must be trained before saving")
        
        try:
            save_dict = {
                'model': self.model,
                'feature_names': self.feature_names_,
                'is_trained': self._is_trained,
                'feature_importances': self._feature_importances,
                'training_metadata': self.training_metadata,
                'parameters': self.get_params(),
                'save_timestamp': datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(save_dict, filepath)
            print(f"Model saved successfully to {filepath}")
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load(cls, filepath: str) -> 'PoultryWeightPredictor':
        """Load a saved model."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        try:
            save_dict = joblib.load(filepath)
            
            # Create new instance with saved parameters
            instance = cls(**save_dict['parameters'])
            
            # Restore saved state
            instance.model = save_dict['model']
            instance.feature_names_ = save_dict['feature_names']
            instance._feature_importances = save_dict['feature_importances']
            instance._is_trained = save_dict['is_trained']
            instance.training_metadata = save_dict['training_metadata']
            
            return instance
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def get_params(self, deep=True) -> Dict:
        """Get parameters (scikit-learn interface)."""
        return {
            'degree': self.degree,
            'fit_intercept': self.fit_intercept,
            'include_bias': self.include_bias
        }
    
    def set_params(self, **parameters) -> 'PoultryWeightPredictor':
        """Set parameters (scikit-learn interface)."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        
        # Update model pipeline with new parameters
        self.model = Pipeline([
            ('poly', PolynomialFeatures(
                degree=self.degree,
                include_bias=self.include_bias
            )),
            ('regressor', LinearRegression(
                fit_intercept=self.fit_intercept
            ))
        ])
        
        return self