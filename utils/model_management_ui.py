import streamlit as st
import pandas as pd
from typing import Dict, List



def render_model_management_ui(model_manager):
    """Render the model management interface with a visually appealing design."""
    # Get list of all models
    models_df = model_manager.list_models()

    if models_df.empty:
        st.info("No saved models found. Train a model to get started!")
        return

    # Format the DataFrame for display
    display_df = models_df.copy()
    display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')

    st.markdown("### **Saved Models**")
    st.markdown("---")

    # Define custom styling for the cards
    st.markdown("""
        <style>
            .card {
                background-color: #f9f9f9;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 10px;
                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .card-header {
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 5px;
                color: #4CAF50;
            }
            .card-content {
                font-size: 14px;
                margin-bottom: 10px;
            }
            .delete-button {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
                cursor: pointer;
                text-align: center;
                transition: all 0.3s ease;
            }
            .delete-button:hover {
                background-color: #c0392b;
            }
        </style>
    """, unsafe_allow_html=True)

    # Render models as individual cards
    for _, row in display_df.iterrows():
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)

            # Card Header
            st.markdown(
                f"""
                <div class="card-header">{row['model_name']}</div>
                """, unsafe_allow_html=True)

            # Card Content
            st.markdown(
                f"""
                <div class="card-content">
                    <b>Type:</b> {row['model_type']}<br>
                    <b>Created At:</b> {row['created_at']}
                </div>
                """, unsafe_allow_html=True)

            # Delete button (Streamlit native)
            delete_key = f"delete_{row['model_id']}"
            if st.button("Delete", key=delete_key):
                try:
                    model_manager.delete_model(row['model_id'])
                    st.success(f"Model '{row['model_name']}' deleted successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting model: {str(e)}")

            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

####################################################################################################################################
def render_model_comparison_interpretation(model_manager, selected_models: List[str]):
    """Render comprehensive model comparison interpretation."""
    st.subheader("📊 Model Comparison Analysis")
    
    if not selected_models:
        st.info("Select models to compare to see the analysis.")
        return
    
    # Get comparison summary
    summary = model_manager.get_comparison_summary(selected_models)
    st.markdown(summary)
    
    # Detailed breakdown
    with st.expander("📈 Detailed Performance Analysis"):
        metrics_comparison = pd.DataFrame([
            model_manager.get_model_metrics(model_id)
            for model_id in selected_models
        ], index=[model_manager.metadata[model_id]['model_name'] for model_id in selected_models])
        
        st.dataframe(metrics_comparison.style.highlight_max(axis=0))
        
        # Add interpretation for each metric
        st.markdown("### Metric Interpretations")
        
        for metric in ['r2', 'mse', 'mae']:
            if metric in metrics_comparison.columns:
                best_model = metrics_comparison[metric].idxmax() if metric == 'r2' else metrics_comparison[metric].idxmin()
                st.markdown(f"**{metric.upper()}**: {best_model} shows the best performance with a value of {metrics_comparison.loc[best_model, metric]:.4f}")
    
    # Training characteristics comparison
    with st.expander("🔍 Training Characteristics"):
        characteristics = []
        for model_id in selected_models:
            meta = model_manager.metadata[model_id]
            chars = meta.get('data_characteristics', {})
            chars['model_name'] = meta['model_name']
            characteristics.append(chars)
        
        chars_df = pd.DataFrame(characteristics)
        if not chars_df.empty:
            st.dataframe(chars_df.set_index('model_name'))
