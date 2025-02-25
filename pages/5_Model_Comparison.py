import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.model_comparison import ModelComparison
from utils.model_manager import ModelManager
from utils.model_management_ui import render_model_management_ui
from models.model_factory import ModelFactory
from utils.auth import check_authentication

def validate_model_data(metadata: dict, model_id: str) -> bool:
    """Validate that required model data is present."""
    required_fields = ['metrics', 'predictions', 'actual']
    missing_fields = [field for field in required_fields if field not in metadata]
    
    if missing_fields:
        st.warning(f"Model {model_id} is missing required data: {', '.join(missing_fields)}")
        return False
    
    # Check for None values
    if metadata['predictions'] is None or metadata['actual'] is None:
        st.warning(f"Model {model_id} has no prediction data. Re-train the model to enable comparison.")
        return False
        
    return True

def app():
    check_authentication()
    st.title("📊 Model Comparison")
    
    # Initialize model manager
    model_manager = ModelManager()
    
    # Get list of saved models
    models_df = model_manager.list_models()
    
    if models_df.empty:
        st.warning("No saved models available for comparison. Please train some models first!")
        return
        
    # Model selection section
    st.subheader("Select Models for Comparison")
    
    # Multi-select for models
    selected_models = st.multiselect(
        "Choose models to compare",
        options=models_df['model_id'].tolist(),
        format_func=lambda x: models_df[models_df['model_id'] == x]['model_name'].iloc[0],
        help="Select two or more models to compare"
    )
    
    if len(selected_models) < 2:
        st.warning("Please select at least two models to enable comparison!")
        return
    
    # Initialize model comparison
    comparison = ModelComparison()
    
    # Try loading the selected models
    valid_models = []
    loaded_data = False
    
    try:
        for model_id in selected_models:
            model, metadata = model_manager.load_model(model_id)
            
            # Validate model data
            if validate_model_data(metadata, model_id):
                # Map feature importance to actual feature names
                feature_importance = metadata.get('feature_importance')
                if feature_importance:
                    # Define the feature name mapping
                    feature_names = ['Int Temp', 'Int Humidity', 'Air Temp', 'Wind Speed', 'Feed Intake']
                    
                    # Create new dictionary with proper feature names
                    mapped_importance = {}
                    for i, (_, value) in enumerate(feature_importance.items()):
                        if i < len(feature_names):
                            mapped_importance[feature_names[i]] = value
                    
                    feature_importance = mapped_importance

                comparison.add_model_results(
                    model_name=metadata['model_name'],
                    metrics=metadata['metrics'],
                    predictions=np.array(metadata['predictions']),
                    actual=np.array(metadata['actual']),
                    feature_importance=feature_importance,
                    metadata=metadata
                )
                valid_models.append(model_id)
                loaded_data = True
        
        if not valid_models:
            st.error("No valid models available for comparison. Please ensure models have prediction data.")
            return
            
        if len(valid_models) < 2:
            st.error("At least two valid models with prediction data are required for comparison.")
            return
        
        st.success(f"Successfully loaded {len(valid_models)} models for comparison")
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.exception(e)
        return

    if loaded_data:
        # Create tabs for different comparisons
        tabs = st.tabs([
            "Overview",
            "Metrics Comparison",
            "Predictions Comparison",
            "Error Analysis",
            "Feature Importance",
            "Model Details"
        ])
        
        # Overview tab
        with tabs[0]:
            st.markdown("## 📊 Model Comparison Overview")
            
            # Get insights
            insights = comparison.generate_comparison_insights()
            
            # Create two columns for the overview
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 🏆 Best Performing Model")
                best_model = insights['overall_best']['model']
                metrics = comparison.get_metrics_comparison()[best_model]
                
                # Display best model metrics in a nice format
                st.markdown(
                    f"""
                    <div style='padding: 10px; border-radius: 5px; background-color: #f0f8ff;'>
                        <h4>{best_model}</h4>
                        <p>Key Metrics:</p>
                        <ul>
                            <li>R² Score: {metrics['r2']:.4f}</li>
                            <li>RMSE: {metrics['rmse']:.4f}</li>
                            <li>MAE: {metrics['mae']:.4f}</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown("### 📈 Models Count")
                st.markdown(
                    f"""
                    <div style='padding: 10px; border-radius: 5px; background-color: #f0fff0;'>
                        <h2 style='text-align: center;'>{len(selected_models)}</h2>
                        <p style='text-align: center;'>Models Compared</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Model Comparison Matrix
            st.markdown("### 📊 Model Performance Matrix")
            metrics_df = comparison.get_metrics_comparison()
            
            # Create a more visually appealing metrics table
            styled_metrics = metrics_df.style.format("{:.4f}").apply(
                lambda x: ['background-color: #90EE90' if (
                    (x.name == 'r2' and v == x.max()) or
                    (x.name in ['mse', 'rmse', 'mae', 'mape'] and v == x.min())
                ) else '' for v in x],
                axis=1
            )
            st.dataframe(styled_metrics, use_container_width=True)
            
            # Quick Insights Section
            st.markdown("### 🔍 Quick Insights")
            
            # Create columns for different insights
            insight_cols = st.columns(3)
            
            with insight_cols[0]:
                st.markdown("#### 🎯 Model Strengths")
                for model, patterns in insights['prediction_patterns'].items():
                    if patterns['reliability'] == 'High':
                        st.markdown(f"- **{model}**: {patterns['bias_tendency']} with high reliability")
            
            with insight_cols[1]:
                st.markdown("#### 🌟 Feature Importance")
                if 'feature_importance' in insights:
                    top_features = list(insights['feature_importance'].get('common_important_features', {}).keys())[:3]
                    if top_features:
                        st.markdown("Top influential features:")
                        for feature in top_features:
                            st.markdown(f"- {feature}")
            
            with insight_cols[2]:
                st.markdown("#### 💡 Recommendations")
                recs = insights.get('recommendations', [])[:3]  # Get top 3 recommendations
                for rec in recs:
                    st.markdown(f"- {rec}")
            
            # Performance Distribution
            st.markdown("### 📉 Performance Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                # R² Score Distribution
                r2_scores = metrics_df.loc['r2']
                fig_r2 = go.Figure(data=[
                    go.Bar(
                        x=[comparison._shorten_model_name(x) for x in r2_scores.index],
                        y=r2_scores.values,
                        marker_color='lightblue'
                    )
                ])
                fig_r2.update_layout(
                    title='R² Score by Model',
                    xaxis_title='Models',
                    yaxis_title='R² Score',
                    xaxis_tickangle=45
                )
                st.plotly_chart(fig_r2, use_container_width=True)
            
            with col2:
                # RMSE Distribution
                rmse_scores = metrics_df.loc['rmse']
                fig_rmse = go.Figure(data=[
                    go.Bar(
                        x=[comparison._shorten_model_name(x) for x in rmse_scores.index],
                        y=rmse_scores.values,
                        marker_color='lightgreen'
                    )
                ])
                fig_rmse.update_layout(
                    title='RMSE by Model',
                    xaxis_title='Models',
                    yaxis_title='RMSE',
                    xaxis_tickangle=45
                )
                st.plotly_chart(fig_rmse, use_container_width=True)
        
        # Metrics Comparison tab
        with tabs[1]:
            st.subheader("Detailed Metrics Comparison")
            
            # Display metrics table
            st.markdown("### Performance Metrics Table")
            formatted_metrics = metrics_df.style.format("{:.4f}").apply(
                lambda x: ['background-color: #e6ffe6' if (
                    (x.name == 'r2' and v == x.max()) or
                    (x.name in ['mse', 'rmse', 'mae', 'mape'] and v == x.min())
                ) else '' for v in x],
                axis=1
            )
            st.dataframe(formatted_metrics, use_container_width=True)
            
            # Metric visualization
            st.markdown("### Metric Visualization")
            metric_to_plot = st.selectbox(
                "Select metric to visualize",
                options=metrics_df.index,
                format_func=lambda x: x.upper()
            )
            
            metrics_plot = comparison.plot_metrics_comparison(metric_to_plot)
            st.plotly_chart(metrics_plot, use_container_width=True)
        
        # Predictions Comparison tab
        with tabs[2]:
            st.subheader("Predictions Analysis")
            
            # Plot predictions comparison
            st.markdown("### Actual vs Predicted Values")
            pred_plot = comparison.plot_prediction_comparison()
            st.plotly_chart(pred_plot, use_container_width=True)
            
            # Show prediction statistics
            st.markdown("### Prediction Statistics")
            stats_df = comparison.get_prediction_stats()
            
            formatted_stats = stats_df.style.format("{:.2f}").apply(
                lambda x: ['background-color: #e6ffe6' if v == x.min() else '' for v in x],
                axis=1
            )
            st.dataframe(formatted_stats, use_container_width=True)
        
        # Error Analysis tab
        with tabs[3]:
            st.subheader("Error Analysis")
            
            error_plot = comparison.plot_error_analysis()
            st.plotly_chart(error_plot, use_container_width=True)
            
            with st.expander("Detailed Error Statistics"):
                error_stats = comparison.get_prediction_stats()
                st.dataframe(
                    error_stats.style.format("{:.4f}")
                    .background_gradient(cmap='RdYlGn_r', subset=['Mean Error', 'Max Error']),
                    use_container_width=True
                )

        # Feature Importance tab
        with tabs[4]:
            st.markdown("## 🎯 Feature Importance Analysis")
            
            importance_plot = comparison.plot_feature_importance_comparison()
            if importance_plot is not None:
                # Add explanation of the heatmap
                st.markdown("""
                ### 📊 Feature Importance Heatmap
                The heatmap below shows the relative importance of each feature across different models.
                - **Darker colors** indicate higher importance
                - **Lighter colors** indicate lower importance
                - Compare across rows to see which features are consistently important
                """)
                
                st.plotly_chart(importance_plot, use_container_width=True)
                
                # Get common important features
                common_features = comparison.get_common_important_features(top_n=3)
                
                # Create two columns for the analysis
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.markdown("### 🔑 Key Features Analysis")
                    st.markdown("##### Most Important Features Across Models:")
                    
                    for rank, (feature, models) in enumerate(common_features.items(), 1):
                        # Create a colored box for each feature with its importance info
                        st.markdown(f"""
                        <div style='padding: 10px; margin: 5px 0; border-radius: 5px; background-color: {"#f0f8ff" if rank % 2 == 0 else "#f5f5f5"}'>
                            <h4 style='margin: 0; color: #1e88e5;'>#{rank} {feature}</h4>
                            <p style='margin: 5px 0;'>Important in {len(models)} models</p>
                            <p style='margin: 0; font-size: 0.9em; color: #666;'>
                                Models: {', '.join([comparison._shorten_model_name(m) for m in models])}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### 📈 Feature Consistency")
                    total_models = len(comparison.results)
                    
                    # Calculate consistency scores
                    for feature, models in common_features.items():
                        consistency = (len(models) / total_models) * 100
                        color = ('green' if consistency >= 75 else 
                                'orange' if consistency >= 50 else 'red')
                        
                        st.markdown(f"""
                        <div style='padding: 8px; margin: 5px 0; border-radius: 5px; border-left: 4px solid {color};'>
                            <p style='margin: 0;'><strong>{feature}</strong></p>
                            <p style='margin: 0; color: {color};'>{consistency:.1f}% consistency</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Add feature importance patterns
                st.markdown("### 🔍 Feature Importance Patterns")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Most stable features
                    st.markdown("#### Most Stable Features")
                    stable_features = comparison.get_stable_features()
                    if stable_features:
                        for feature, stability_score in stable_features.items():
                            st.markdown(f"""
                            <div style='padding: 5px; margin: 2px 0;'>
                                <strong>{feature}</strong>: {stability_score:.2f} stability score
                            </div>
                            """, unsafe_allow_html=True)
                
                with col2:
                    # Feature importance recommendations
                    st.markdown("#### Recommendations")
                    st.markdown("""
                    - Focus on consistently important features
                    - Consider feature engineering for top features
                    - Monitor less important features for potential removal
                    """)
            else:
                st.info("Feature importance comparison not available for these models. This might be because the selected models don't provide feature importance information.")
                
            # Add download button for feature importance data
            if importance_plot is not None:
                st.markdown("### 📥 Export Feature Importance Data")
                # Get feature importance data
                importance_data = {}
                for model_name, result in comparison.results.items():
                    if result['feature_importance'] is not None:
                        importance_data[comparison._shorten_model_name(model_name)] = result['feature_importance']
                
                if importance_data:
                    importance_df = pd.DataFrame(importance_data)
                    csv = importance_df.to_csv(index=True)
                    st.download_button(
                        label="Download Feature Importance Data",
                        data=csv,
                        file_name="feature_importance.csv",
                        mime="text/csv"
                    )                

        
        # Model Details tab
        with tabs[5]:
            st.subheader("Model Details")
            
            for model_id in valid_models:
                metadata = model_manager.metadata[model_id]
                with st.expander(f"Model: {metadata.get('model_name', 'Unknown Model')}"):
                    st.write("**General Information**")
                    st.write(f"- Model Type: {metadata.get('model_type', 'Not specified')}")
                    st.write(f"- Version: {metadata.get('version', 'Not specified')}")
                    if 'training_date' in metadata:
                        st.write(f"- Training Date: {metadata['training_date']}")
                    
                    if 'model_params' in metadata:
                        st.write("\n**Model Parameters**")
                        for param, value in metadata['model_params'].items():
                            st.write(f"- {param}: {value}")
                    
                    if 'metrics' in metadata:
                        st.write("\n**Performance Metrics**")
                        for metric, value in metadata['metrics'].items():
                            if isinstance(value, (int, float)):
                                st.write(f"- {metric}: {value:.4f}")
                            else:
                                st.write(f"- {metric}: {value}")
                    
                    if 'data_characteristics' in metadata:
                        st.write("\n**Data Characteristics**")
                        chars = metadata['data_characteristics']
                        if 'total_samples' in chars:
                            st.write(f"- Total Samples: {chars['total_samples']}")
                        if 'features' in chars:
                            st.write(f"- Number of Features: {chars['features']}")
                        if 'data_size' in chars:
                            st.write(f"- Data Size: {chars['data_size']}")
                        if 'has_outliers' in chars:
                            st.write(f"- Contains Outliers: {chars['has_outliers']}")
                    
                    if 'training_config' in metadata:
                        st.write("\n**Training Configuration**")
                        config = metadata['training_config']
                        if 'test_size' in config:
                            st.write(f"- Test Size: {config['test_size']}")
                        if 'cross_validation' in config:
                            st.write(f"- Cross Validation: {config['cross_validation']}")
                        if config.get('cv_folds'):
                            st.write(f"- CV Folds: {config['cv_folds']}")
                        if 'parameters' in config:
                            st.write("- Model Parameters:")
                            for param, value in config['parameters'].items():
                                st.write(f"  • {param}: {value}")
        
        # Export options
        st.sidebar.subheader("Export Options")
        if st.sidebar.button("Generate Report"):
            report = comparison.export_comparison_report()
            
            # Convert report to Excel
            output = pd.ExcelWriter('model_comparison_report.xlsx', engine='xlsxwriter')
            
            # Write each component
            report['metrics_comparison'].to_excel(output, sheet_name='Metrics')
            report['prediction_comparison'].to_excel(output, sheet_name='Predictions')
            
            # Export error statistics
            stats_df = comparison.get_prediction_stats()
            stats_df.to_excel(output, sheet_name='Error Statistics')
            
            output.close()
            
            # Offer download
            with open('model_comparison_report.xlsx', 'rb') as f:
                st.sidebar.download_button(
                    label="Download Report",
                    data=f,
                    file_name="model_comparison_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    app()