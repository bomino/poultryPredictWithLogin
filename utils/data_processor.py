import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define constants
REQUIRED_COLUMNS = [
    'Int Temp',
    'Int Humidity',
    'Air Temp',
    'Wind Speed',
    'Feed Intake',
    'Weight'
]

FEATURE_COLUMNS = [
    'Int Temp',
    'Int Humidity',
    'Air Temp',
    'Wind Speed',
    'Feed Intake'
]

TARGET_COLUMN = 'Weight'
RANDOM_STATE = 42

class DataProcessor:
    def __init__(self):
        """Initialize the DataProcessor with a standard scaler."""
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def validate_data(self, df: pd.DataFrame, is_training: bool = True) -> None:
        """
        Validate the input dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe to validate
            is_training (bool): Whether this is training data (requires 2+ rows) or prediction data (1+ rows)
        """
        if df is None:
            raise ValueError("DataFrame is None")
        if df.empty:
            raise ValueError("DataFrame is empty")
        if df[FEATURE_COLUMNS].isnull().any().any():
            raise ValueError("DataFrame contains null values in feature columns")
        if is_training and len(df) < 2:
            raise ValueError("Training data must contain at least 2 rows")
        if not is_training and len(df) < 1:
            raise ValueError("Prediction data must contain at least 1 row")
            
    def validate_columns(self, df: pd.DataFrame) -> tuple[bool, list]:
        """Validate if all required columns are present in the dataframe."""
        missing_cols = []
        # For prediction, we only need feature columns
        required_cols = REQUIRED_COLUMNS if 'Weight' in df.columns else FEATURE_COLUMNS
        missing_cols = [col for col in required_cols if col not in df.columns]
        return len(missing_cols) == 0, missing_cols
    
    def preprocess_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Preprocess the input dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe to preprocess
            is_training (bool): Whether this is training data or prediction data
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        # Validate input
        self.validate_data(df, is_training=is_training)
        
        # Create a copy
        df = df.copy()
        
        print("\nPreprocessing Data:")
        print(f"Initial shape: {df.shape}")
        
        # Convert data types
        columns_to_process = REQUIRED_COLUMNS if is_training else FEATURE_COLUMNS
        for col in columns_to_process:
            try:
                print(f"\nProcessing column: {col}")
                # Handle string values
                if df[col].dtype == 'object':
                    df[col] = df[col].str.strip()
                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"Column {col} converted successfully")
                print(f"Sample values: {df[col].head().tolist()}")
            except Exception as e:
                print(f"Error converting column {col}: {str(e)}")
                raise ValueError(f"Error converting column {col}: {str(e)}")
        
        # Remove rows with any null values
        initial_rows = len(df)
        df = df.dropna()
        rows_dropped = initial_rows - len(df)
        print(f"\nRows dropped due to null values: {rows_dropped}")
        
        # Validate after preprocessing
        if len(df) == 0:
            raise ValueError("No valid data remaining after preprocessing")
        
        # Fit the scaler on features if training
        if is_training:
            print("\nFitting scaler on features...")
            self.scaler.fit(df[FEATURE_COLUMNS])
            self.is_fitted = True
            print("Scaler fitted successfully")
        elif not self.is_fitted:
            raise ValueError("Scaler must be fitted before preprocessing prediction data")
        
        print(f"Final shape: {df.shape}")
        return df
    
    def prepare_features(self, df: pd.DataFrame, test_size: float = 0.2) -> tuple:
        """Prepare features for model training."""
        # Validate input
        self.validate_data(df, is_training=True)
        
        # Select features and target
        X = df[FEATURE_COLUMNS]
        y = df[TARGET_COLUMN]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE
        )
        
        # Ensure scaler is fitted
        if not self.is_fitted:
            self.scaler.fit(X_train)
            self.is_fitted = True
        
        # Scale features
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def scale_features(self, X: pd.DataFrame) -> np.ndarray:
        """Scale features using the fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted yet. Run preprocess_data first.")
        
        if isinstance(X, pd.DataFrame):
            missing_cols = [col for col in FEATURE_COLUMNS if col not in X.columns]
            if missing_cols:
                raise ValueError(f"Missing required feature columns: {missing_cols}")
            X = X[FEATURE_COLUMNS]
        
        return self.scaler.transform(X)
    
    @staticmethod
    def calculate_statistics(df: pd.DataFrame) -> dict:
        """Calculate basic statistics for the dataset."""
        if df.empty:
            raise ValueError("Cannot calculate statistics on empty DataFrame")
            
        stats = {}
        columns = [col for col in REQUIRED_COLUMNS if col in df.columns]
        for col in columns:
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median()
            }
        return stats
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, column: str) -> pd.Series:
        """Detect outliers using IQR method."""
        if df.empty:
            raise ValueError("Cannot detect outliers in empty DataFrame")
            
        if column not in df.columns:
            raise ValueError(f"Column {column} not found")
            
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (df[column] < lower_bound) | (df[column] > upper_bound)