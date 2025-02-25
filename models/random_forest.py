from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import numpy as np
from config.settings import MODEL_SAVE_PATH

class PoultryRandomForest:
    def __init__(self, **kwargs):
        """
        Initialize the Random Forest model with parameters.
        """
        # Default parameters
        self.default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'auto',
            'bootstrap': True,
            'n_jobs': -1,
            'random_state': 42,
            'oob_score': False
        }
        
        # Update defaults with provided parameters
        self.params = {**self.default_params, **kwargs}
        
        # Initialize the model with parameters
        self.model = RandomForestRegressor(**self.params)
        self._is_trained = False
        self.feature_names = None
        
    @property
    def is_trained(self):
        """Check if the model is trained."""
        return self._is_trained
        
    def set_feature_names(self, feature_names):
        """Set feature names for the model."""
        self.feature_names = feature_names
        
    def train(self, X_train, y_train, feature_names=None):
        """Train the model."""
        if X_train is None or y_train is None:
            raise ValueError("Training data cannot be None")
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Training data cannot be empty")
            
        try:
            print("Training Random Forest model...")
            print(f"Training data shapes: X={X_train.shape}, y={y_train.shape}")
            print("Model parameters:", self.params)
            
            if feature_names is not None:
                self.feature_names = feature_names
            
            self.model.fit(X_train, y_train)
            self._is_trained = True
            print("Model trained successfully")
            
            if hasattr(self.model, 'feature_importances_'):
                print("Feature importances available")
            
            if self.params['oob_score']:
                print(f"OOB Score: {self.model.oob_score_:.4f}")
                
            return self
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
            
    def predict(self, X):
        """Make predictions using the trained model."""
        if not self._is_trained:
            raise ValueError("Model needs to be trained before making predictions")
        
        if X is None or len(X) == 0:
            raise ValueError("Input data cannot be empty")
            
        try:
            return self.model.predict(X)
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model performance."""
        if not self._is_trained:
            raise ValueError("Model needs to be trained before evaluation")
            
        try:
            # Make predictions
            y_pred = self.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': mean_squared_error(y_test, y_pred, squared=False),
                'r2': r2_score(y_test, y_pred),
                'mae': np.mean(np.abs(y_test - y_pred)),
                'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            }
            
            # Add OOB score if available
            if hasattr(self.model, 'oob_score_'):
                metrics['oob_score'] = self.model.oob_score_
            
            return metrics, y_pred
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            raise
    
    def get_feature_importance(self):
        """Get feature importance based on the trained model."""
        if not self._is_trained:
            raise ValueError("Model needs to be trained before getting feature importance")
            
        try:
            if not hasattr(self.model, 'feature_importances_'):
                raise ValueError("Model does not support feature importance")
                
            # Get feature importances
            importances = self.model.feature_importances_
            
            # Use stored feature names if available, otherwise use indices
            feature_names = self.feature_names if self.feature_names is not None else [f'Feature_{i}' for i in range(len(importances))]
            
            # Create feature importance dictionary
            importance_dict = dict(zip(feature_names, importances))
            
            # Sort by importance
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            print(f"Error getting feature importance: {str(e)}")
            raise
    
    def save(self, filepath):
        """Save the model to a file."""
        if not self._is_trained:
            raise ValueError("Model needs to be trained before saving")
            
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            # Save the entire object
            joblib.dump(self, filepath)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load(cls, filepath):
        """Load a model from a file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        try:
            # Load the model
            model = joblib.load(filepath)
            if not isinstance(model, cls):
                raise ValueError("Loaded file is not a valid model")
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def get_params(self):
        """Get the current model parameters."""
        return self.params
    
    def set_params(self, **params):
        """Update model parameters."""
        self.params.update(params)
        self.model.set_params(**self.params)
        self._is_trained = False  # Reset trained status since parameters changed