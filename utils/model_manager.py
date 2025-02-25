import os
import json
from datetime import datetime
import joblib
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

class ModelManager:
    def __init__(self, models_dir: str = "models/saved_models"):
        """Initialize ModelManager with models directory."""
        self.models_dir = models_dir
        self.metadata_file = os.path.join(models_dir, "model_metadata.json")
        self._ensure_directory_exists()
        self.metadata = self._load_metadata()

    def _ensure_directory_exists(self):
        """Create models directory if it doesn't exist."""
        os.makedirs(self.models_dir, exist_ok=True)

    def _load_metadata(self) -> Dict:
        """Load model metadata from JSON file."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _convert_numpy_types(self, obj):
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (str, int, float)):
            return obj
        elif obj is None:
            return None
        else:
            return str(obj)

    def _save_metadata(self):
        """Save model metadata to JSON file."""
        # Convert all metadata to JSON-serializable format
        serializable_metadata = self._convert_numpy_types(self.metadata)
        with open(self.metadata_file, 'w') as f:
            json.dump(serializable_metadata, f, indent=4)

    def save_model(self, model_name: str, model: Any, metadata: Dict) -> str:
        """
        Save model and its metadata.
        
        Args:
            model_name: Name of the model
            model: The model instance
            metadata: Dictionary containing model metadata
        Returns:
            model_id (str)
        """
        model_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = os.path.join(self.models_dir, f"{model_id}.joblib")
        
        # Ensure predictions and actual values are included
        if 'predictions' not in metadata or 'actual' not in metadata:
            raise ValueError("Model metadata must include 'predictions' and 'actual' values")
        
        # Convert all metadata to JSON-serializable format
        metadata = self._convert_numpy_types(metadata)
            
        # Save model file
        joblib.dump(model, model_path)
        
        # Update metadata
        self.metadata[model_id] = {
            **metadata,
            'model_path': model_path,
            'created_at': datetime.now().isoformat(),
            'model_name': model_name
        }
        
        try:
            self._save_metadata()
        except Exception as e:
            # If saving metadata fails, delete the model file
            if os.path.exists(model_path):
                os.remove(model_path)
            raise Exception(f"Failed to save metadata: {str(e)}")
        
        return model_id

    def load_model(self, model_id: str) -> tuple[Any, Dict]:
        """Load model and its metadata."""
        if model_id not in self.metadata:
            raise ValueError(f"Model {model_id} not found")
            
        model_path = self.metadata[model_id]['model_path']
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model = joblib.load(model_path)
        return model, self.metadata[model_id]

    def delete_model(self, model_id: str):
        """Delete model and its metadata."""
        if model_id not in self.metadata:
            raise ValueError(f"Model {model_id} not found")
            
        # Delete model file
        model_path = self.metadata[model_id]['model_path']
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
                print(f"Deleted model file: {model_path}")
            except Exception as e:
                print(f"Failed to delete model file: {str(e)}")
        else:
            print(f"Model file not found: {model_path}")
            
        # Remove metadata
        del self.metadata[model_id]
        self._save_metadata()
        print(f"Deleted model metadata for: {model_id}")


    def list_models(self) -> pd.DataFrame:
        """Return DataFrame with model information."""
        if not self.metadata:
            return pd.DataFrame()
            
        models_list = []
        for model_id, meta in self.metadata.items():
            model_info = {
                'model_id': model_id,
                'model_name': meta['model_name'],
                'created_at': meta['created_at'],
                'model_type': meta.get('model_type', 'Unknown'),
                'performance_metrics': meta.get('metrics', {}),
                'data_characteristics': meta.get('data_characteristics', {})
            }
            models_list.append(model_info)
            
        return pd.DataFrame(models_list)

    def get_model_metrics(self, model_id: str) -> Dict:
        """Get model performance metrics."""
        if model_id not in self.metadata:
            raise ValueError(f"Model {model_id} not found")
        return self.metadata[model_id].get('metrics', {})

    def get_comparison_summary(self, model_ids: List[str]) -> str:
        """Generate natural language comparison summary."""
        if not model_ids:
            return "No models selected for comparison."
            
        summaries = []
        metrics = {}
        
        # Collect metrics for all models
        for model_id in model_ids:
            model_metrics = self.get_model_metrics(model_id)
            model_name = self.metadata[model_id]['model_name']
            metrics[model_name] = model_metrics
        
        # Find best model for each metric
        best_models = {}
        for metric in ['r2', 'mse', 'mae']:
            if metric in next(iter(metrics.values())):
                if metric == 'r2':
                    best_model = max(metrics.items(), key=lambda x: x[1].get(metric, 0))
                    best_models[metric] = best_model[0]
                else:
                    best_model = min(metrics.items(), key=lambda x: x[1].get(metric, float('inf')))
                    best_models[metric] = best_model[0]
        
        # Generate summary
        summaries.append(f"Comparing {len(model_ids)} models:")
        if 'r2' in best_models:
            summaries.append(f"• {best_models['r2']} shows the best overall performance with highest R² score")
        if 'mse' in best_models:
            summaries.append(f"• {best_models['mse']} achieves the lowest Mean Squared Error")
        if 'mae' in best_models:
            summaries.append(f"• {best_models['mae']} has the best Mean Absolute Error")
        
        return "\n".join(summaries)