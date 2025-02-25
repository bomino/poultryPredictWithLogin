

# App settings
APP_NAME = "Poultry Weight Predictor"
APP_ICON = "🐔"
LAYOUT = "wide"

# Data settings
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

# Model settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
POLYNOMIAL_DEGREE = 2

# File paths
MODEL_SAVE_PATH = "models/saved_models"
TEMP_DATA_PATH = "temp/data"

# Visualization settings
THEME_COLORS = {
    'primary': '#FF4B4B',
    'secondary': '#0083B8',
    'background': '#FFFFFF',
    'text': '#262730'
}

PLOT_HEIGHT = 500
PLOT_WIDTH = 800