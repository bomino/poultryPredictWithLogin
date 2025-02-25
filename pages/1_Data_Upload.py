import streamlit as st
import pandas as pd
from utils.data_processor import DataProcessor
from utils.visualizations import Visualizer
from utils.auth import check_authentication

def app():
    check_authentication()
    st.title("📤 Data Upload and Preview")
    
    # Initialize objects
    data_processor = DataProcessor()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your CSV file", 
        type=['csv'],
        help="Upload a CSV file containing poultry data"
    )
    
    if uploaded_file is not None:
        try:
            # Debug information
            st.write("File uploaded successfully")
            st.write(f"Filename: {uploaded_file.name}")
            
            # Read the data
            df = pd.read_csv(uploaded_file)
            
            # Debug information
            st.write(f"Data shape: {df.shape}")
            st.write("Columns found:", df.columns.tolist())
            
            # Display raw data preview before processing
            st.subheader("Raw Data Preview")
            st.dataframe(df.head())
            
            # Validate columns
            is_valid, missing_cols = data_processor.validate_columns(df)
            
            if not is_valid:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                st.stop()
                
            # Display column information
            st.subheader("Column Information")
            st.write(df.dtypes)
            
            # Show number of null values
            st.subheader("Null Values Count")
            st.write(df.isnull().sum())
            
            # Process the data
            df_processed = data_processor.preprocess_data(df)
            
            # Debug information
            st.write(f"Processed data shape: {df_processed.shape}")
            
            # Display processed data preview
            st.subheader("Processed Data Preview")
            st.dataframe(df_processed.head())
            
            # Save to session state
            st.session_state['data'] = df_processed
            
            # Display basic statistics
            st.subheader("Basic Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Number of Records", len(df_processed))
                st.metric("Number of Features", len(df_processed.columns)-1)
            
            with col2:
                st.metric("Missing Values", df_processed.isnull().sum().sum())
                st.metric("Duplicate Records", df_processed.duplicated().sum())
            
            # Display summary statistics
            st.subheader("Summary Statistics")
            st.write(df_processed.describe())
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Error details:", e)
            import traceback
            st.write("Traceback:", traceback.format_exc())
    
    else:
        st.info("Please upload a CSV file to begin.")
        
        # Show sample data format
        st.subheader("Required Data Format")
        sample_data = pd.DataFrame({
            'Int Temp': [30.5, 31.2, 29.8],
            'Int Humidity': [65, 68, 70],
            'Air Temp': [28.5, 29.0, 27.5],
            'Wind Speed': [4.2, 3.8, 4.5],
            'Feed Intake': [150, 155, 148],
            'Weight': [1200, 1250, 1180]
        })
        
        st.write("Your CSV file should have these columns:")
        st.dataframe(sample_data)
        
        # Download sample template
        st.download_button(
            label="Download Sample Template",
            data=sample_data.to_csv(index=False),
            file_name="sample_template.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    app()