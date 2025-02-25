import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from utils.data_processor import DataProcessor, FEATURE_COLUMNS
from models.polynomial_regression import PoultryWeightPredictor
from config.settings import MODEL_SAVE_PATH
from utils.auth import check_authentication 

def validate_input_values(input_values: dict) -> bool:
    """Validate input values for manual prediction."""
    for key, value in input_values.items():
        if value is None or pd.isna(value):
            st.error(f"Invalid value for {key}")
            return False
    return True

def app():
    check_authentication()
    st.title("🔮 Make Predictions")
    
    # Initialize data processor
    data_processor = DataProcessor()
    
    # Sidebar - Model Selection
    st.sidebar.subheader("Model Selection")
    
    # Option to use trained model or load saved model
    model_option = st.sidebar.radio(
        "Choose Model Source",
        ["Use Currently Trained Model", "Load Saved Model"]
    )
    
    # Model loading section
    try:
        if model_option == "Use Currently Trained Model":
            if 'model' not in st.session_state:
                st.error("No trained model found! Please train a model first.")
                st.stop()
            model = st.session_state['model']
            
            # Use the data processor from training if available
            if 'data_processor' in st.session_state:
                data_processor = st.session_state['data_processor']
                print("Using data processor from training session")
        else:
            # Load saved model
            if not os.path.exists(MODEL_SAVE_PATH):
                st.error(f"Model directory not found: {MODEL_SAVE_PATH}")
                st.stop()
                
            saved_models = [f for f in os.listdir(MODEL_SAVE_PATH) if f.endswith('.joblib')]
            
            if not saved_models:
                st.error("No saved models found!")
                st.stop()
                
            selected_model = st.sidebar.selectbox("Select Saved Model", saved_models)
            try:
                model_path = os.path.join(MODEL_SAVE_PATH, selected_model)
                saved_data = joblib.load(model_path)
                
                # Check if it's a dictionary containing model and data_processor
                if isinstance(saved_data, dict):
                    model = saved_data['model']
                    data_processor = saved_data['data_processor']
                else:
                    model = saved_data
                
                if not isinstance(model, PoultryWeightPredictor):
                    st.error("Invalid model file!")
                    st.stop()
                
                st.sidebar.success(f"Model loaded successfully: {selected_model}")
                
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.stop()
                
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        st.stop()
    
    # Main content
    st.subheader("Make New Predictions")
    
    # Input method selection
    input_method = st.radio(
        "Choose Input Method",
        ["Manual Input", "Batch Prediction (CSV)"]
    )
    
    if input_method == "Manual Input":
        st.markdown("### Enter Feature Values")
        
        try:
            # Create input fields for each feature
            input_values = {}
            col1, col2 = st.columns(2)
            
            with col1:
                input_values['Int Temp'] = st.number_input(
                    "Internal Temperature (°C)",
                    min_value=0.0,
                    max_value=50.0,
                    value=25.0,
                    step=0.1,
                    help="Temperature inside the poultry house"
                )
                input_values['Int Humidity'] = st.number_input(
                    "Internal Humidity (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=60.0,
                    step=1.0,
                    help="Humidity level inside the poultry house"
                )
                input_values['Air Temp'] = st.number_input(
                    "Air Temperature (°C)",
                    min_value=0.0,
                    max_value=50.0,
                    value=23.0,
                    step=0.1,
                    help="Outside air temperature"
                )
            
            with col2:
                input_values['Wind Speed'] = st.number_input(
                    "Wind Speed (m/s)",
                    min_value=0.0,
                    max_value=20.0,
                    value=2.0,
                    step=0.1,
                    help="Wind speed measurement"
                )
                input_values['Feed Intake'] = st.number_input(
                    "Feed Intake (g)",
                    min_value=0.0,
                    value=100.0,
                    step=1.0,
                    help="Amount of feed consumed"
                )
            
            if st.button("Predict"):
                try:
                    # Validate input values
                    if not validate_input_values(input_values):
                        st.stop()
                    
                    # Create DataFrame from input
                    input_df = pd.DataFrame([input_values])
                    
                    # Show input data
                    st.write("Input Data:")
                    st.dataframe(input_df)
                    
                    # Scale features directly (no need for preprocessing for single prediction)
                    scaled_features = data_processor.scale_features(input_df[FEATURE_COLUMNS])
                    
                    # Make prediction
                    prediction = model.predict(scaled_features)
                    
                    # Display prediction
                    st.success(f"Predicted Weight: {prediction[0]:.2f} g")
                    
                    # Add to prediction history
                    if 'prediction_history' not in st.session_state:
                        st.session_state['prediction_history'] = []
                    
                    st.session_state['prediction_history'].append({
                        **input_values,
                        'Predicted Weight': prediction[0]
                    })
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        except Exception as e:
            st.error(f"Error setting up input fields: {str(e)}")
    
    else:  # Batch Prediction
        st.markdown("### Upload Data for Batch Prediction")
        
        # Show sample format
        st.info("Your CSV file should have these columns: " + ", ".join(FEATURE_COLUMNS))
        
        # Sample data
        sample_df = pd.DataFrame({
            'Int Temp': [25.0, 26.0],
            'Int Humidity': [60.0, 65.0],
            'Air Temp': [23.0, 24.0],
            'Wind Speed': [2.0, 2.5],
            'Feed Intake': [100.0, 110.0]
        })
        
        # Show and download sample template
        st.write("Sample Format:")
        st.dataframe(sample_df)
        st.download_button(
            "Download Sample Template",
            sample_df.to_csv(index=False),
            "sample_template.csv",
            "text/csv"
        )
        
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload a CSV file containing the required features"
        )
        
        if uploaded_file is not None:
            try:
                # Read data
                prediction_df = pd.read_csv(uploaded_file)
                
                # Show raw data
                st.subheader("Input Data Preview")
                st.dataframe(prediction_df.head())
                
                # Validate columns
                is_valid, missing_cols = data_processor.validate_columns(prediction_df)
                
                if not is_valid:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    st.write("Required columns:", FEATURE_COLUMNS)
                    st.stop()
                
                # Scale features
                scaled_features = data_processor.scale_features(prediction_df[FEATURE_COLUMNS])
                
                # Make predictions
                predictions = model.predict(scaled_features)
                
                # Create results DataFrame
                results_df = prediction_df.copy()
                results_df['Predicted_Weight'] = predictions
                
                # Display results
                st.subheader("Prediction Results")
                st.dataframe(results_df)
                
                # Show statistics
                st.subheader("Prediction Statistics")
                stats = results_df['Predicted_Weight'].describe()
                st.write(stats)
                
                # Download results
                st.download_button(
                    label="Download Predictions",
                    data=results_df.to_csv(index=False),
                    file_name="predictions.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Show prediction history
    if 'prediction_history' in st.session_state and st.session_state['prediction_history']:
        st.subheader("Prediction History")
        history_df = pd.DataFrame(st.session_state['prediction_history'])
        st.dataframe(history_df)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear History"):
                st.session_state['prediction_history'] = []
                st.rerun()
        with col2:
            st.download_button(
                label="Download History",
                data=history_df.to_csv(index=False),
                file_name="prediction_history.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    app()