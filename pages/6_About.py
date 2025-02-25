import streamlit as st
from utils.auth import check_authentication

def app():
    st.title("ℹ️ About")
    
    st.markdown("""
    # Poultry Weight Predictor

    This application helps poultry farmers and researchers predict poultry weight based on environmental 
    and feeding data using advanced machine learning techniques and comprehensive model comparison capabilities.

    ## Features

    1. **Data Upload and Analysis**
       - Upload CSV files with poultry data
       - Automatic data validation and preprocessing
       - Interactive data visualization
       - Comprehensive statistical analysis
       - Multi-level outlier detection and analysis
       - Advanced data quality assessment

    2. **Advanced Analytics**
       - Time series analysis of weight progression
       - Feature relationship exploration
       - Multi-dimensional correlation analysis
       - Pattern detection and visualization
       - Comprehensive outlier analysis across features
       - Interactive data exploration tools
                
      3. **Intelligent Model Selection**
       - Automated model recommendations based on:
         * Dataset size and characteristics
         * Presence of outliers
         * Data complexity
         * Feature relationships
       - Detailed reasoning for each recommendation
       - Alternative model suggestions
       - Parameter optimization guidance
       - Performance expectations
       - Use case considerations

    4. **Machine Learning Models**
       - Multiple model support with intelligent selection:
         * Polynomial Regression (recommended for small, clean datasets)
         * Gradient Boosting (recommended for large, complex datasets)
         * Support Vector Regression (recommended for datasets with outliers)
         * Random Forest (recommended for balanced performance and interpretability)
       - Automated feature importance analysis
       - Model persistence and versioning
       - Cross-validation capabilities
       - Early stopping for appropriate models
       - Smart parameter suggestions

    5. **Model Training and Evaluation**
       - Interactive parameter tuning
       - Real-time performance metrics
       - Feature importance visualization
       - Advanced error analysis
       - Model saving and loading functionality
       - Comprehensive training metadata tracking

    6. **Predictions**
       - Single prediction through manual input
       - Batch predictions through CSV upload
       - Prediction history tracking
       - Confidence intervals
       - Downloadable prediction results
       - Performance monitoring
       - Prediction validation

    7. **Model Comparison**
       - Side-by-side model performance comparison
       - Comparative metrics visualization
       - Feature importance comparison across models
       - Prediction accuracy analysis
       - Detailed performance metrics:
         * Mean Squared Error (MSE)
         * Root Mean Squared Error (RMSE)
         * R² Score
         * Mean Absolute Error (MAE)
         * Mean Absolute Percentage Error (MAPE)
       - Exportable comparison reports
       - Visual performance charts
       - Best model recommendation based on data characteristics

    ## How to Use

    1. **Data Upload**: Start by uploading your CSV file in the Data Upload page
    2. **Data Analysis**: Explore and analyze your data with interactive visualizations
    3. **Model Training**: Train different models with optimized parameters
    4. **Model Comparison**: Compare models to select the best performer
    5. **Predictions**: Make predictions using your chosen model
    6. **Export Results**: Download predictions and comparison reports

    ## Data Requirements

    Your input data should contain the following features:
    - Internal Temperature (°C)
    - Internal Humidity (%)
    - Air Temperature (°C)
    - Wind Speed (m/s)
    - Feed Intake (g)
    - Weight (g) - required for training data only

    ## Model Details

    ### Polynomial Regression
    - Captures non-linear relationships
    - Good for baseline predictions
    - Highly interpretable results
    - Efficient with smaller datasets
    - Perfect for understanding basic patterns

    ### Gradient Boosting
    - Handles complex patterns
    - High prediction accuracy
    - Robust feature importance
    - Excellent for large datasets
    - Support for early stopping

    ### Support Vector Regression
    - Robust to outliers
    - Excellent generalization
    - Handles non-linear relationships
    - Kernel-based learning
    - Perfect for medium-sized datasets
                
      ### Random Forest
    - Excellent balance of performance and interpretability
    - Built-in feature importance
    - Robust to outliers and noise
    - Provides uncertainty estimates
    - No feature scaling required
    - Great for medium to large datasets            

    ### Model Comparison Capabilities
    - Automated performance metric calculation
    - Visual comparison tools
    - Feature importance analysis
    - Prediction accuracy comparison
    - Export functionality for detailed reports
    - Best model selection assistance
    - Cross-model validation

    ## Technical Details

    - Built with Streamlit for interactive web interface
    - Scikit-learn for machine learning models
    - Plotly for interactive visualizations
    - Pandas for efficient data manipulation
    - Advanced error handling and validation
    - Comprehensive model comparison framework
    - Robust data processing pipeline

    ## Data Security

    - Local data processing
    - No data storage without user consent
    - Secure model saving and loading
    - Privacy-focused design
    - Transparent data handling

    ## Support

    For support, feature requests, or bug reports, please contact:
    - Email: Bomino@mlawali.com
    - GitHub: [Project Repository](https://github.com/bomino/PoultryPredict3)

    ## Version Information

    - Current Version: 2.1.0
    - Last Updated: November 2024
    - Key Features: 
      * Multi-model support with Random Forest
      * Enhanced model comparison capabilities
      * Improved feature importance analysis
      * Advanced uncertainty estimation
      * Comprehensive cross-validation
      * Better parameter optimization
      * Enhanced error handling
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit by Bomino")

if __name__ == "__main__":
    app()