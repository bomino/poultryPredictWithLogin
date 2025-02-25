import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from config.settings import THEME_COLORS, PLOT_HEIGHT, PLOT_WIDTH

class Visualizer:
    @staticmethod
    def plot_correlation_matrix(df: pd.DataFrame):
        """Create a correlation matrix heatmap."""
        corr = df.corr()
        fig = px.imshow(
            corr,
            color_continuous_scale='RdBu',
            title='Feature Correlation Matrix'
        )
        fig.update_layout(
            height=PLOT_HEIGHT,
            width=PLOT_WIDTH
        )
        return fig
    
    @staticmethod
    def plot_feature_importance(feature_names: list, importance_values: list):
        """Create a feature importance bar plot."""
        fig = px.bar(
            x=importance_values,
            y=feature_names,
            orientation='h',
            title='Feature Importance',
            color=importance_values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            height=PLOT_HEIGHT,
            width=PLOT_WIDTH,
            yaxis_title='Features',
            xaxis_title='Importance'
        )
        return fig
    
    @staticmethod
    def plot_actual_vs_predicted(y_true: list, y_pred: list):
        """Create scatter plot of actual vs predicted values."""
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=y_true,
            y=y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(
                color=THEME_COLORS['secondary'],
                size=8
            )
        ))
        
        # Add perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(
                color=THEME_COLORS['primary'],
                dash='dash'
            )
        ))
        
        fig.update_layout(
            title='Actual vs Predicted Values',
            xaxis_title='Actual Weight',
            yaxis_title='Predicted Weight',
            height=PLOT_HEIGHT,
            width=PLOT_WIDTH
        )
        
        return fig

    @staticmethod
    def plot_residuals(y_true: list, y_pred: list):
        """Create residuals plot with shape validation."""
        # Convert to numpy arrays and ensure same length
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        residuals = y_true - y_pred
        fig = go.Figure()

        # Add residuals scatter plot
        fig.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(
                color=THEME_COLORS['secondary'],
                size=8
            )
        ))

        # Add reference lines
        fig.add_hline(y=0, line_dash="dash", line_color=THEME_COLORS['primary'])
        
        mean_residual = np.mean(residuals)
        fig.add_hline(
            y=mean_residual,
            line_dash="dot",
            line_color="red",
            name=f'Mean Residual ({mean_residual:.2f})'
        )

        # Add standard error bands
        std_residual = np.std(residuals)
        fig.add_hline(
            y=mean_residual + 2*std_residual,
            line_dash="dot",
            line_color="gray",
            name='+2 Std Dev'
        )
        fig.add_hline(
            y=mean_residual - 2*std_residual,
            line_dash="dot",
            line_color="gray",
            name='-2 Std Dev'
        )

        fig.update_layout(
            title='Residuals Analysis',
            xaxis_title='Predicted Values',
            yaxis_title='Residuals (Actual - Predicted)',
            height=PLOT_HEIGHT,
            width=PLOT_WIDTH,
            showlegend=True,
            hovermode='closest'
        )

        return fig
    
    @staticmethod
    def plot_weight_over_time(df: pd.DataFrame):
        """Create line plot of weight progression over time."""
        fig = px.line(
            df,
            y='Weight',
            title='Weight Progression Over Time',
            markers=True
        )
        
        fig.update_layout(
            height=PLOT_HEIGHT,
            width=PLOT_WIDTH,
            xaxis_title='Time Points',
            yaxis_title='Weight'
        )
        
        return fig
    
    @staticmethod
    def plot_feature_distribution(df: pd.DataFrame, column: str):
        """Create histogram of feature distribution."""
        fig = px.histogram(
            df,
            x=column,
            title=f'Distribution of {column}',
            color_discrete_sequence=[THEME_COLORS['secondary']]
        )
        
        fig.update_layout(
            height=PLOT_HEIGHT,
            width=PLOT_WIDTH,
            xaxis_title=column,
            yaxis_title='Count'
        )
        
        return fig