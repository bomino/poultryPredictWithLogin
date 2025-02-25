from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

class PoultrySVR(BaseEstimator, RegressorMixin):
    """Support Vector Regression model for poultry weight prediction."""
    
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1, gamma='scale',
                 cache_size=200, max_iter=-1, random_state=42):
        """
        Initialize the SVR model with parameters.
        
        Args:
            kernel: Kernel type ('rbf', 'linear', 'poly')
            C: Regularization parameter
            epsilon: Epsilon in epsilon-SVR model
            gamma: Kernel coefficient
            cache_size: Kernel cache size in MB
            max_iter: Maximum iterations
            random_state: Random state for reproducibility
        """
        self.kernel = kernel
        self.C = float(C)
        self.epsilon = float(epsilon)
        self.gamma = gamma
        self.cache_size = int(cache_size)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        
        # Initialize model and scaler
        self.model = None
        self.scaler = StandardScaler()
        self._is_trained = False
        self.feature_names_ = None
        self.training_metadata = {}
    
    def _get_model_params(self) -> Dict:
        """Get parameters for SVR model."""
        return {
            'kernel': self.kernel,
            'C': self.C,
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'cache_size': self.cache_size,
            'max_iter': self.max_iter
        }
        
    def fit(self, X, y):
        """Fit method for scikit-learn compatibility."""
        return self.train(X, y)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              feature_names: Optional[List[str]] = None) -> 'PoultrySVR':
        """
        Train the SVR model with scaled features.
        """
        try:
            # Store feature names if provided
            if feature_names is not None:
                self.feature_names_ = feature_names
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X_train)
            
            # Initialize and train model
            self.model = SVR(**self._get_model_params())
            self.model.fit(X_scaled, y_train)
            
            # Mark as trained
            self._is_trained = True
            
            # Store training metadata
            self.training_metadata = {
                'n_samples': len(X_train),
                'n_features': X_train.shape[1],
                'n_support_vectors': len(self.model.support_vectors_),
                'training_date': datetime.now().isoformat(),
                'parameters': self._get_model_params()
            }
            
            return self
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if not self._is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        """Evaluate model performance."""
        if not self._is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        try:
            y_pred = self.predict(X_test)
            
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': mean_squared_error(y_test, y_pred, squared=False),
                'r2': r2_score(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            }
            
            return metrics, y_pred
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            raise
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get feature importance scores.
        Note: SVR doesn't provide direct feature importance, so we use coefficient magnitudes for linear kernel
        and provide uniform importance for other kernels.
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before getting feature importance")
            
        try:
            if feature_names is None:
                feature_names = self.feature_names_ or [f'feature_{i}' for i in range(len(self.scaler.mean_))]
            
            # For linear kernel, use coefficient magnitudes
            if self.kernel == 'linear':
                importances = np.abs(self.model.coef_[0])
            else:
                # For non-linear kernels, provide uniform importance
                importances = np.ones(len(feature_names)) / len(feature_names)
            
            # Create and sort importance dictionary
            importance_dict = dict(zip(feature_names, importances))
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
                'scaler': self.scaler,
                'feature_names': self.feature_names_,
                'is_trained': self._is_trained,
                'training_metadata': self.training_metadata,
                'parameters': self._get_model_params(),
                'save_timestamp': datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(save_dict, filepath)
            print(f"Model saved successfully to {filepath}")
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load(cls, filepath: str) -> 'PoultrySVR':
        """Load a saved model."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        try:
            save_dict = joblib.load(filepath)
            
            # Create new instance with saved parameters
            instance = cls(**save_dict['parameters'])
            
            # Restore saved state
            instance.model = save_dict['model']
            instance.scaler = save_dict['scaler']
            instance.feature_names_ = save_dict['feature_names']
            instance._is_trained = save_dict['is_trained']
            instance.training_metadata = save_dict['training_metadata']
            
            return instance
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def get_params(self, deep=True) -> Dict:
        """Get parameters (scikit-learn interface)."""
        return {
            'kernel': self.kernel,
            'C': self.C,
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'cache_size': self.cache_size,
            'max_iter': self.max_iter,
            'random_state': self.random_state
        }
    
    def set_params(self, **parameters) -> 'PoultrySVR':
        """Set parameters (scikit-learn interface)."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self