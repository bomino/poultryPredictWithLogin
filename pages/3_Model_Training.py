import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import traceback
import hashlib
from datetime import datetime
from sklearn.model_selection import cross_val_score, cross_validate
from typing import Dict, Tuple, List, Any
from utils.data_processor import DataProcessor, FEATURE_COLUMNS
from utils.visualizations import Visualizer
from models.model_factory import ModelFactory
from config.settings import MODEL_SAVE_PATH
from utils.model_manager import ModelManager
from utils.model_management_ui import render_model_management_ui
from utils.auth import check_authentication 

VERSION = "2.0.0"

def generate_data_hash(df: pd.DataFrame) -> str:
    """Generate a hash of the training data for versioning."""
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()

def analyze_data_characteristics(df: pd.DataFrame, data_processor: DataProcessor) -> Dict:
    """Analyze data characteristics for model selection and insights."""
    
    # Check for outliers in each feature column
    outliers_by_column = {
        col: data_processor.detect_outliers(df, col).any()
        for col in FEATURE_COLUMNS
    }
    
    characteristics = {
        'n_samples': len(df),
        'n_features': len(FEATURE_COLUMNS),
        'has_outliers': any(outliers_by_column.values()),
        'outliers_by_column': outliers_by_column,
        'missing_values': df.isnull().sum().sum(),
        'feature_correlations': df[FEATURE_COLUMNS].corr(),
        'data_size': 'large' if len(df) > 1000 else 'medium' if len(df) > 100 else 'small',
        'complexity': 'high' if len(df) > 1000 else 'medium',
        'feature_stats': {col: {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'skew': df[col].skew(),
            'kurtosis': df[col].kurtosis(),
            'has_outliers': outliers_by_column[col]
        } for col in FEATURE_COLUMNS}
    }
    
    return characteristics

def validate_model_params(params: Dict, model_type: str, model_factory: ModelFactory) -> Tuple[bool, str]:
    """Enhanced parameter validation for model parameters."""
    param_info = model_factory.get_model_params(model_type)
    
    for param_name, value in params.items():
        if param_name not in param_info:
            return False, f"Unexpected parameter: {param_name}"
            
        if isinstance(value, (int, float)):
            if value <= 0:
                return False, f"Parameter {param_name} must be positive"
    return True, ""

def perform_cross_validation(model: Any, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
    """Perform cross-validation and return detailed scores."""
    try:
        scoring = {
            'r2': 'r2',
            'mse': 'neg_mean_squared_error',
            'mae': 'neg_mean_absolute_error'
        }
        
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
        
        return {
            'cv_r2_mean': scores['test_r2'].mean(),
            'cv_r2_std': scores['test_r2'].std(),
            'cv_mse_mean': -scores['test_mse'].mean(),
            'cv_mse_std': scores['test_mse'].std(),
            'cv_mae_mean': -scores['test_mae'].mean(),
            'cv_mae_std': scores['test_mae'].std()
        }
    except Exception as e:
        st.warning(f"Cross-validation failed: {str(e)}")
        return {}

def get_model_recommendation(data_characteristics: Dict) -> Tuple[str, str, List[str]]:
    """Get model recommendations based on data characteristics."""
    reasons = []
    
    if data_characteristics['data_size'] == 'small':
        if not data_characteristics['has_outliers']:
            return 'polynomial', "Polynomial Regression", [
                "Small dataset size",
                "No significant outliers",
                "Good for interpretability"
            ]
    
    if data_characteristics['has_outliers']:
        reasons.append("Dataset contains outliers")
        
    if data_characteristics['data_size'] in ['medium', 'large']:
        reasons.append("Sufficient data for complex patterns")
        
    return 'gradient_boosting', "Gradient Boosting", reasons


def app():
    check_authentication()
    st.title("🎯 Model Training")
    
    # Initialize model manager
    model_manager = ModelManager()
    
    # Check if data exists in session state
    if 'data' not in st.session_state:
        st.error("Please upload data in the Data Upload page first!")
        st.stop()
    
    # Initialize objects
    data_processor = DataProcessor()
    visualizer = Visualizer()
    model_factory = ModelFactory()
    
    # Get and process data
    df = st.session_state['data']
    
    try:
        # Process data and analyze characteristics
        df_processed = data_processor.preprocess_data(df)
        data_characteristics = analyze_data_characteristics(df_processed, data_processor)
        data_hash = generate_data_hash(df_processed)
        
        # Save data_processor in session state
        st.session_state['data_processor'] = data_processor
        
        # Success message
        st.success(f"Data preprocessed successfully: {df_processed.shape[0]} rows")
        
        # Data Overview Section
        st.subheader("📊 Data Overview")
        
        # Basic metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", data_characteristics['n_samples'])
            st.metric("Features", data_characteristics['n_features'])
        with col2:
            st.metric("Missing Values", data_characteristics['missing_values'])
            st.metric("Contains Outliers", "Yes" if data_characteristics['has_outliers'] else "No")
        with col3:
            st.metric("Data Size", data_characteristics['data_size'].title())
            st.metric("Complexity", data_characteristics['complexity'].title())
        
        # Outlier Information
        if data_characteristics['has_outliers']:
            st.markdown("#### Outlier Information")
            st.write("**Features with Outliers:**")
            outlier_cols = [col for col, has_outliers in 
                        data_characteristics['outliers_by_column'].items() 
                        if has_outliers]
            
            # Create columns for outlier information
            n_cols = 3
            cols = st.columns(n_cols)
            for idx, col in enumerate(outlier_cols):
                with cols[idx % n_cols]:
                    st.write(f"• {col}")
                    stats = data_characteristics['feature_stats'][col]
                    st.write(f"  Mean: {stats['mean']:.2f}")
                    st.write(f"  Std: {stats['std']:.2f}")
        
        # Feature Statistics
        with st.expander("Feature Statistics"):
            stats_df = pd.DataFrame({
                col: {
                    'Mean': stats['mean'],
                    'Std Dev': stats['std'],
                    'Skewness': stats['skew'],
                    'Kurtosis': stats['kurtosis']
                }
                for col, stats in data_characteristics['feature_stats'].items()
            }).T
            
            st.dataframe(stats_df.style.format("{:.2f}"))
        
        # Feature Correlations
        with st.expander("Feature Correlations"):
            st.plotly_chart(
                visualizer.plot_correlation_matrix(
                    data_characteristics['feature_correlations']
                ),
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        st.code(traceback.format_exc())
        st.stop()

            #####################################
    # Get model recommendation
    recommended_model, rec_name, rec_reasons = get_model_recommendation(data_characteristics)
    
    # Sidebar - Model Selection with Recommendation
    st.sidebar.subheader("Model Selection")
    available_models = model_factory.get_available_models()
    
    selected_model_type = st.sidebar.selectbox(
        "Select Model Type",
        list(available_models.keys()),
        index=list(available_models.keys()).index(recommended_model),
        format_func=lambda x: f"{available_models[x]['name']} {'✨ (Recommended)' if x == recommended_model else ''}"
    )
    
    # Show recommendation reasoning
    if recommended_model == selected_model_type:
        st.sidebar.success(f"✨ Recommended model based on:")
        for reason in rec_reasons:
            st.sidebar.write(f"- {reason}")
    else:
        st.sidebar.info(f"💡 {rec_name} is recommended for your data based on its characteristics.")
    
    # Model Information
    with st.sidebar.expander("Model Details", expanded=True):
        st.write(f"**{available_models[selected_model_type]['name']}**")
        st.write(available_models[selected_model_type]['description'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Strengths:**")
            for strength in available_models[selected_model_type]['strengths']:
                st.write(f"✓ {strength}")
        with col2:
            st.write("**Limitations:**")
            for limitation in available_models[selected_model_type]['limitations']:
                st.write(f"• {limitation}")
    
    # Training Configuration
    st.sidebar.subheader("Training Configuration")
    
    # Advanced settings toggle
    show_advanced = st.sidebar.checkbox("Show Advanced Settings", value=False)
    
    # Model parameters
    default_params = model_factory.get_model_params(selected_model_type)
    param_descriptions = model_factory.get_param_descriptions(selected_model_type)
    
    # Model parameters
    model_params = {}
    with st.sidebar.expander("Model Parameters", expanded=True):
        for param, param_info in default_params.items():
            # Skip random_state for all models except gradient_boosting
            if param == 'random_state' and selected_model_type != 'gradient_boosting':
                continue
                
            if param in ['subsample', 'min_samples_split'] and not show_advanced:
                continue

            if 'type' in param_info:
                if param_info['type'] == 'int':
                    model_params[param] = st.number_input(
                        param,
                        min_value=1,
                        value=param_info['default'],
                        help=param_descriptions[param]
                    )
                elif param_info['type'] == 'float':
                    # Different ranges for different parameters
                    min_val = param_info.get('range', (0.0, 1.0))[0]
                    max_val = param_info.get('range', (0.0, 1.0))[1]
                    step = 0.01 if param == 'learning_rate' else 0.1
                    
                    model_params[param] = st.slider(
                        param,
                        min_value=min_val,
                        max_value=max_val,
                        value=param_info['default'],
                        step=step,
                        format="%.3f",
                        help=param_descriptions[param]
                    )
                elif param_info['type'] == 'select':
                    model_params[param] = st.selectbox(
                        param,
                        options=param_info['options'],
                        index=param_info['options'].index(param_info['default']),
                        help=param_descriptions[param]
                    )

    # Add fixed parameters
    if selected_model_type == 'gradient_boosting':
        model_params['random_state'] = 42
    
    # Training settings
    st.sidebar.subheader("Training Settings")
    
    # Test size slider
    min_test_size = max(0.1, 1 / len(df_processed))
    test_size = st.sidebar.slider(
        "Test Set Size", 
        min_value=min_test_size,
        max_value=0.4,
        value=0.2,
        step=0.05,
        help="Proportion of dataset to include in the test split"
    )
    
    # Advanced training settings
    if show_advanced:
        with st.sidebar.expander("Advanced Training Settings"):
            use_cv = st.checkbox("Use Cross-validation", value=True)
            if use_cv:
                cv_folds = st.slider("Number of CV Folds", min_value=2, max_value=10, value=5)
            else:
                cv_folds = 0
            
            if selected_model_type == 'gradient_boosting':
                use_early_stopping = st.checkbox("Use Early Stopping", value=True)
                if use_early_stopping:
                    early_stopping_rounds = st.slider(
                        "Early Stopping Rounds",
                        min_value=5,
                        max_value=50,
                        value=10
                    )
                    model_params['early_stopping_rounds'] = early_stopping_rounds
    else:
        use_cv = False
        cv_folds = 0
    
    # Add fixed parameters
    model_params['random_state'] = 42

    ################################################################################

    # Main content
    st.subheader("Data Split and Model Training")
    
    # Show feature information
    with st.expander("Feature Information", expanded=True):
        st.write("Features being used:", FEATURE_COLUMNS)
        
        # Feature correlations heatmap
        if st.checkbox("Show Feature Correlations"):
            corr_matrix = data_characteristics['feature_correlations']
            st.plotly_chart(visualizer.plot_correlation_matrix(corr_matrix), use_container_width=True)
    
    # Prepare features
    try:
        X_train, X_test, y_train, y_test = data_processor.prepare_features(
            df_processed, 
            test_size=test_size
        )
        st.success("Features prepared successfully")
        
        # Data split information
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Training set shape: {X_train.shape}")
        with col2:
            st.write(f"Test set shape: {X_test.shape}")
        
    except Exception as e:
        st.error(f"Error preparing features: {str(e)}")
        st.code(traceback.format_exc())
        st.stop()
    
    # Save test data
    st.session_state['test_data'] = {
        'X_test': X_test,
        'y_test': y_test
    }
   ################################################################################# 
    # Training section
    training_container = st.container()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create model instance
    model = model_factory.get_model(selected_model_type, model_params)
    
    # Train model button
    if st.button("Train Model", key='train_button'):
        try:
            # Training phase
            status_text.text("Training model...")
            progress_bar.progress(20)
            
        
            feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else FEATURE_COLUMNS

             # Train model
            if hasattr(model, 'set_feature_names'):
                model.set_feature_names(feature_names)
                model.train(X_train, y_train)
            else:
                model.train(X_train, y_train, feature_names=feature_names)
            progress_bar.progress(40)

            
            # Cross-validation if enabled
            cv_results = {}
            if use_cv:
                status_text.text("Performing cross-validation...")
                cv_results = perform_cross_validation(model, X_train, y_train, cv_folds)
                progress_bar.progress(60)
            
            # Model evaluation
            status_text.text("Evaluating model...")
            metrics, y_pred = model.evaluate(X_test, y_test)
            progress_bar.progress(80)
            
            # Feature importance
            importance_dict = model.get_feature_importance()
            progress_bar.progress(90)
            
            # Create model metadata
            metadata = {
                'version': VERSION,
                'training_date': datetime.now().isoformat(),
                'data_hash': data_hash,
                'model_type': selected_model_type,
                'model_params': model_params,
                'feature_columns': FEATURE_COLUMNS,
                'metrics': metrics,
                'cv_results': cv_results,
                'feature_importance': importance_dict,
                'data_characteristics': {
                    'total_samples': len(df_processed),
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'features': len(FEATURE_COLUMNS),
                    'has_outliers': data_characteristics['has_outliers'],
                    'data_size': data_characteristics['data_size'],
                    'complexity': data_characteristics['complexity']
                },
                'training_config': {
                    'test_size': test_size,
                    'cross_validation': use_cv,
                    'cv_folds': cv_folds if use_cv else None,
                    'parameters': model_params
                }
            }
            
            # Store in ModelManager
            model_id = model_manager.save_model(
                        model_name=f"{available_models[selected_model_type]['name']}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
                        model=model,
                        metadata={
                            'version': VERSION,
                            'model_type': selected_model_type,
                            'metrics': metrics,
                            'predictions': y_pred.tolist(),  # Include predictions
                            'actual': y_test.tolist(),       # Include actual values
                            'data_characteristics': data_characteristics,
                            'training_config': {
                                'test_size': test_size,
                                'cross_validation': use_cv,
                                'cv_folds': cv_folds if use_cv else None,
                                'parameters': model_params
                            },
                            'feature_importance': importance_dict,
                            'data_hash': data_hash
                        }
                    )
            
            # Store in session state
            st.session_state['model'] = model
            st.session_state['model_metadata'] = metadata
            st.session_state['training_results'] = {
                'metrics': metrics,
                'predictions': y_pred,
                'feature_importance': importance_dict,
                'cv_results': cv_results,
                'test_size': test_size
            }
            
            progress_bar.progress(100)
            status_text.text("Training completed successfully!")
            st.success(f"Model trained and saved successfully with ID: {model_id}")
            
        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            st.code(traceback.format_exc())
            progress_bar.empty()
            status_text.empty()

            ############################################################################

            # Display results if available
    if st.session_state.get('training_results'):
        results = st.session_state['training_results']
        metrics = results['metrics']
        y_pred = results['predictions']
        importance_dict = results['feature_importance']
        cv_results = results.get('cv_results', {})
        
        # Create tabs for different sections
        tabs = st.tabs(["Model Performance", "Cross-Validation", "Predictions", "Feature Importance", "Model Management"])
        
        # Model Performance tab
        with tabs[0]:
            st.subheader("Model Performance Metrics")
            
            # Basic metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Squared Error", f"{metrics['mse']:.2f}")
            with col2:
                st.metric("Root MSE", f"{metrics['rmse']:.2f}")
            with col3:
                st.metric("R² Score", f"{metrics['r2']:.4f}")
            
            # Additional metrics if available
            if 'mae' in metrics:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Absolute Error", f"{metrics['mae']:.2f}")
                with col2:
                    st.metric("Mean Absolute % Error", f"{metrics.get('mape', 0):.2f}%")
            
            # Performance visualization
            st.plotly_chart(visualizer.plot_residuals(y_test, y_pred), use_container_width=True)
        
        # Cross-Validation tab
        with tabs[1]:
            st.subheader("Cross-Validation Results")
            if cv_results:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("CV R² Score", f"{cv_results['cv_r2_mean']:.4f} ± {cv_results['cv_r2_std']:.4f}")
                    st.metric("CV MSE", f"{cv_results['cv_mse_mean']:.2f} ± {cv_results['cv_mse_std']:.2f}")
                with col2:
                    st.metric("CV MAE", f"{cv_results['cv_mae_mean']:.2f} ± {cv_results['cv_mae_std']:.2f}")
            else:
                st.info("Cross-validation was not performed. Enable it in Advanced Settings to see CV results.")

        # Predictions tab
        with tabs[2]:
            st.subheader("Actual vs Predicted Values")
            
            # Main prediction plot
            prediction_plot = visualizer.plot_actual_vs_predicted(
                st.session_state['test_data']['y_test'],
                y_pred
            )
            st.plotly_chart(prediction_plot, use_container_width=True)
            
            # Detailed predictions section
            with st.expander("Detailed Predictions Analysis", expanded=False):
                # Summary statistics
                errors = np.abs(y_test - y_pred)
                rel_errors = (errors / y_test) * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average Error", f"{np.mean(errors):.2f}")
                    st.metric("Max Error", f"{np.max(errors):.2f}")
                with col2:
                    st.metric("Average % Error", f"{np.mean(rel_errors):.2f}%")
                    st.metric("Max % Error", f"{np.max(rel_errors):.2f}%")
                
                # Detailed predictions table
                if st.checkbox("Show detailed predictions"):
                    n_examples = st.slider("Number of examples", min_value=5, max_value=len(y_pred), value=10)
                    examples = pd.DataFrame({
                        'Actual Weight': st.session_state['test_data']['y_test'][:n_examples],
                        'Predicted Weight': y_pred[:n_examples],
                        'Absolute Error': errors[:n_examples],
                        'Relative Error (%)': rel_errors[:n_examples]
                    })
                    st.dataframe(examples.style.highlight_max(axis=0))
                    
                    # Download predictions
                    st.download_button(
                        label="Download All Predictions",
                        data=examples.to_csv(index=False),
                        file_name="model_predictions.csv",
                        mime="text/csv"
                    )
        
        # Feature Importance tab
        with tabs[3]:
            st.subheader("Feature Importance Analysis")
            
            # Feature importance plot
            importance_plot = visualizer.plot_feature_importance(
                list(importance_dict.keys()),
                list(importance_dict.values())
            )
            st.plotly_chart(importance_plot, use_container_width=True)
            
            # Detailed feature analysis
            with st.expander("Feature Importance Details"):
                # Create and sort importance DataFrame
                importance_df = pd.DataFrame({
                    'Feature': importance_dict.keys(),
                    'Importance': importance_dict.values()
                }).sort_values('Importance', ascending=False)
                
                # Add cumulative importance
                importance_df['Cumulative Importance'] = importance_df['Importance'].cumsum()
                
                # Display table with formatting
                st.dataframe(
                    importance_df.style.format({
                        'Importance': '{:.3f}',
                        'Cumulative Importance': '{:.3f}'
                    }).bar(subset=['Importance'], color='#0083B8')
                )
                
                # Feature importance insights
                st.write("**Key Insights:**")
                top_features = importance_df.head(3)['Feature'].tolist()
                st.write(f"• Top 3 most important features: {', '.join(top_features)}")
                cumulative_80 = len(importance_df[importance_df['Cumulative Importance'] <= 0.8])
                st.write(f"• {cumulative_80} features account for 80% of the model's predictions")

        # Model Management tab
        with tabs[4]:
            st.subheader("Model Management")
            render_model_management_ui(model_manager)
            
            # Model metadata section
            if st.session_state.get('model_metadata'):
                with st.expander("Current Model Metadata", expanded=True):
                    metadata = st.session_state['model_metadata']
                    
                    st.write("**General Information**")
                    st.write(f"- Model Type: {metadata['model_type']}")
                    st.write(f"- Version: {metadata['version']}")
                    st.write(f"- Training Date: {metadata['training_date']}")
                    
                    st.write("\n**Model Parameters**")
                    for param, value in metadata['model_params'].items():
                        st.write(f"- {param}: {value}")
                    
                    st.write("\n**Performance Metrics**")
                    for metric, value in metadata['metrics'].items():
                        st.write(f"- {metric}: {value:.4f}")

if __name__ == "__main__":
    app()
