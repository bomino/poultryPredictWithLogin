from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

class PoultryGBRegressor(BaseEstimator, RegressorMixin):
    """
    Gradient Boosting Regressor for poultry weight prediction with enhanced functionality.
    Includes early stopping, feature importance analysis, and model persistence.
    Implements scikit-learn's estimator interface.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 min_samples_split=2, min_samples_leaf=1, subsample=1.0,
                 random_state=42, early_stopping_rounds=None, validation_fraction=0.1):
        """
        Initialize the Gradient Boosting model.
        
        Args:
            n_estimators: Number of boosting stages
            learning_rate: Learning rate shrinks the contribution of each tree
            max_depth: Maximum depth of the individual trees
            min_samples_split: Minimum samples required to split an internal node
            min_samples_leaf: Minimum samples required to be at a leaf node
            subsample: Fraction of samples to be used for fitting the individual trees
            random_state: Random state for reproducibility
            early_stopping_rounds: Number of rounds with no improvement to stop training
            validation_fraction: Fraction of training data to use for validation
        """
        # Convert parameters to correct types
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.subsample = float(subsample)
        self.random_state = int(random_state)
        self.early_stopping_rounds = int(early_stopping_rounds) if early_stopping_rounds is not None else None
        self.validation_fraction = float(validation_fraction)
        
        # Initialize state variables
        self.model = None
        self._is_trained = False
        self.feature_names_ = None
        self._feature_importances = None
        self.training_metadata = {}

    def _get_model_params(self) -> Dict:
            """Get parameters for scikit-learn model."""
            return {
                'n_estimators': int(self.n_estimators),  # Ensure int
                'learning_rate': float(self.learning_rate),
                'max_depth': int(self.max_depth),
                'min_samples_split': int(self.min_samples_split),
                'min_samples_leaf': int(self.min_samples_leaf),
                'subsample': float(self.subsample),
                'random_state': int(self.random_state)
            }


    def _validate_input_data(self, X: np.ndarray, y: Optional[np.ndarray] = None, is_training: bool = True):
        """
        Validate input data for training or prediction.
        
        Args:
            X: Input features
            y: Target values (optional)
            is_training: Whether this is for training data
        """
        if X is None:
            raise ValueError("Input features cannot be None")
        if len(X) == 0:
            raise ValueError("Input features cannot be empty")
        if is_training:
            if y is None:
                raise ValueError("Target values cannot be None for training")
            if len(X) != len(y):
                raise ValueError("Number of samples in features and target must match")
            if len(X) < 2:
                raise ValueError("Need at least 2 samples for training")

    def fit(self, X, y):
        """
        Fit the model (scikit-learn interface).
        
        Args:
            X: Training features
            y: Target values
            
        Returns:
            self: The trained model instance
        """
        return self.train(X, y)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              feature_names: Optional[List[str]] = None) -> 'PoultryGBRegressor':
        """
        Train the model with optional early stopping.
        
        Args:
            X_train: Training features
            y_train: Training targets
            feature_names: Optional list of feature names
        """
        try:
            # Validate input data
            self._validate_input_data(X_train, y_train, is_training=True)
            
            # Convert to numpy arrays
            X_train = np.asarray(X_train)
            y_train = np.asarray(y_train)
            
            # Store feature names if provided
            if feature_names is not None:
                self.feature_names_ = feature_names
            
            print(f"Training Gradient Boosting model with {len(X_train)} samples...")
            
            # Implementation of early stopping if enabled
            if self.early_stopping_rounds is not None:
                self._train_with_early_stopping(X_train, y_train)
            else:
                # Initialize base model with explicit parameters
                self.model = GradientBoostingRegressor(
                    n_estimators=int(self.n_estimators),
                    learning_rate=float(self.learning_rate),
                    max_depth=int(self.max_depth),
                    min_samples_split=int(self.min_samples_split),
                    min_samples_leaf=int(self.min_samples_leaf),
                    subsample=float(self.subsample),
                    random_state=int(self.random_state)
                )
                self.model.fit(X_train, y_train)
            
            # Store feature importances and mark as trained
            self._feature_importances = self.model.feature_importances_
            self._is_trained = True
            
            # Store training metadata
            self.training_metadata = {
                'n_samples': len(X_train),
                'n_features': X_train.shape[1],
                'training_date': datetime.now().isoformat(),
                'parameters': self._get_model_params(),
                'early_stopping_used': self.early_stopping_rounds is not None
            }
            
            print("Model training completed successfully")
            return self
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise




    def _train_with_early_stopping(self, X_train: np.ndarray, y_train: np.ndarray):
            """
            Implement early stopping training procedure.
            
            Args:
                X_train: Training features
                y_train: Training targets
            """
            try:
                # Split data for validation
                n_samples = len(X_train)
                n_val = int(n_samples * self.validation_fraction)
                indices = np.random.permutation(n_samples)
                val_indices = indices[:n_val]
                train_indices = indices[n_val:]
                
                # Split the data
                X_train_sub = X_train[train_indices]
                y_train_sub = y_train[train_indices]
                X_val = X_train[val_indices]
                y_val = y_train[val_indices]
                
                best_val_score = float('-inf')
                best_model = None
                patience_counter = 0
                
                print(f"Starting early stopping training with {len(X_train_sub)} training samples and {len(X_val)} validation samples")
                
                # Get base parameters and remove non-GradientBoostingRegressor parameters
                base_params = self._get_model_params()
                
                for n_est in range(1, self.n_estimators + 1):
                    # Create model with current number of estimators
                    current_model = GradientBoostingRegressor(
                        n_estimators=n_est,
                        learning_rate=self.learning_rate,
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        min_samples_leaf=self.min_samples_leaf,
                        subsample=self.subsample,
                        random_state=self.random_state
                    )
                    
                    # Train the model
                    current_model.fit(X_train_sub, y_train_sub)
                    
                    # Evaluate on validation set
                    val_score = current_model.score(X_val, y_val)
                    
                    # Print progress every 10 iterations
                    if n_est % 10 == 0:
                        print(f"Iteration {n_est}: validation score = {val_score:.4f}")
                    
                    if val_score > best_val_score:
                        best_val_score = val_score
                        best_model = current_model
                        patience_counter = 0
                        print(f"New best validation score: {best_val_score:.4f} at iteration {n_est}")
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= self.early_stopping_rounds:
                        print(f"Early stopping triggered at iteration {n_est} (no improvement for {self.early_stopping_rounds} rounds)")
                        break
                
                if best_model is None:
                    raise ValueError("Training failed to produce a valid model")
                    
                self.model = best_model
                print(f"Training completed. Best validation score: {best_val_score:.4f}")
                
            except Exception as e:
                print(f"Error in early stopping training: {str(e)}")
                raise


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Predicted values
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self._validate_input_data(X, is_training=False)
        return self.model.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Evaluate model performance with multiple metrics.
        
        Args:
            X_test: Test features
            y_test: True test values
            
        Returns:
            Tuple containing:
                - Dictionary of evaluation metrics
                - Array of predicted values
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        try:
            self._validate_input_data(X_test, y_test, is_training=False)
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
        
        Args:
            feature_names: Optional list of feature names to use
            
        Returns:
            Dict[str, float]: Dictionary mapping feature names to importance scores
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        try:
            # Use provided feature names, stored names, or generate default names
            if feature_names is None:
                feature_names = self.feature_names_
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(self._feature_importances))]
            
            # Create and sort importance dictionary
            importance_dict = dict(zip(feature_names, self._feature_importances))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            print(f"Error getting feature importance: {str(e)}")
            raise

    def save(self, filepath: str):
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path where to save the model
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before saving")
        
        try:
            save_dict = {
                'model': self.model,
                'params': self.get_params(),
                'feature_names': self.feature_names_,
                'feature_importances': self._feature_importances,
                'is_trained': self._is_trained,
                'training_metadata': self.training_metadata,
                'save_timestamp': datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(save_dict, filepath)
            print(f"Model saved successfully to {filepath}")
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            raise

    @classmethod
    def load(cls, filepath: str) -> 'PoultryGBRegressor':
        """
        Load a saved model from a file.
        
        Args:
            filepath: Path to the saved model file
            
        Returns:
            PoultryGBRegressor: Loaded model instance
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        try:
            save_dict = joblib.load(filepath)
            
            # Create new instance with saved parameters
            instance = cls(**save_dict['params'])
            
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

    @property
    def feature_importances_(self):
        """Get feature importances (scikit-learn interface)."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self._feature_importances

    @property
    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        return self._is_trained

    def get_params(self, deep=True) -> Dict:
        """Get parameters (scikit-learn interface)."""
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'subsample': self.subsample,
            'random_state': self.random_state,
            'early_stopping_rounds': self.early_stopping_rounds,
            'validation_fraction': self.validation_fraction
        }

    def set_params(self, **parameters) -> 'PoultryGBRegressor':
        """Set parameters (scikit-learn interface)."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self