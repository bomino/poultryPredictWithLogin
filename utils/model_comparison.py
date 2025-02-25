import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class ModelComparison:
    def __init__(self):
        """Initialize the ModelComparison class with enhanced tracking."""
        self.results = {}
        self.predictions = {}
        self.model_metadata = {}
        self.metric_descriptions = {
            'r2': 'Coefficient of determination (higher is better)',
            'mse': 'Mean squared error (lower is better)',
            'rmse': 'Root mean squared error (lower is better)',
            'mae': 'Mean absolute error (lower is better)',
            'mape': 'Mean absolute percentage error (lower is better)'
        }
    
    def _shorten_model_name(self, name: str) -> str:
        """
        Shorten model name for better display in graphs.
        Example: 'Gradient Boosting_20241126_121613' -> 'GB_121613'
        """
        # Dictionary of common model name mappings
        model_mappings = {
            'Gradient Boosting': 'GB',
            'Polynomial Regression': 'PR',
            'Support Vector': 'SVR'
        }
        
        # Split into name and timestamp
        if '_' in name:
            base_name, timestamp = name.split('_', 1)
            
            # Shorten base name if it's in our mappings
            for full_name, short_name in model_mappings.items():
                if full_name in base_name:
                    base_name = short_name
                    break
            
            # Keep only the time part of the timestamp (last 6 digits)
            if len(timestamp) >= 6:
                timestamp = timestamp[-6:]
                
            return f"{base_name}_{timestamp}"
        
        return name


    def add_model_results(self, model_name: str, metrics: Dict[str, float], 
                         predictions: np.ndarray, actual: np.ndarray,
                         feature_importance: Dict[str, float] = None,
                         metadata: Dict = None):
        """
        Add results for a model to the comparison.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of performance metrics
            predictions: Array of predicted values
            actual: Array of actual values
            feature_importance: Dictionary of feature importance scores
            metadata: Additional model metadata
        """
        # Store basic results
        self.results[model_name] = {
            'metrics': metrics,
            'feature_importance': feature_importance
        }
        
        # Store predictions
        self.predictions[model_name] = {
            'predicted': np.array(predictions),
            'actual': np.array(actual)
        }
        
        # Store metadata
        self.model_metadata[model_name] = metadata or {}
        
        # Calculate additional metrics
        errors = np.abs(actual - predictions)
        rel_errors = np.abs((actual - predictions) / actual) * 100
        
        additional_metrics = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'error_90th_percentile': np.percentile(errors, 90),
            'error_95th_percentile': np.percentile(errors, 95)
        }
        
        # Update metrics with additional ones
        self.results[model_name]['metrics'].update(additional_metrics)
    
    def get_metrics_comparison(self) -> pd.DataFrame:
        """Get a DataFrame comparing metrics across models with enhanced formatting."""
        metrics_data = {}
        for model_name, result in self.results.items():
            metrics_data[model_name] = result['metrics']
        
        metrics_df = pd.DataFrame(metrics_data).round(4)
        
        # Sort metrics in a logical order
        metric_order = [
            'r2', 'mse', 'rmse', 'mae', 'mape',
            'mean_error', 'std_error', 'max_error', 'min_error',
            'error_90th_percentile', 'error_95th_percentile'
        ]
        
        # Reorder metrics if they exist
        available_metrics = [m for m in metric_order if m in metrics_df.index]
        other_metrics = [m for m in metrics_df.index if m not in metric_order]
        ordered_metrics = available_metrics + other_metrics
        
        return metrics_df.loc[ordered_metrics]
    #############################################################

    def get_prediction_comparison(self) -> pd.DataFrame:
        """Get a comprehensive DataFrame comparing predictions across models."""
        if not self.predictions:
            return pd.DataFrame()
        
        # First, create a base DataFrame with the actual values
        first_model = list(self.predictions.keys())[0]
        base_df = pd.DataFrame({
            'Sample': range(1, len(self.predictions[first_model]['actual']) + 1),
            'Actual': self.predictions[first_model]['actual']
        })
        
        # Add predictions and comprehensive error metrics for each model
        for model_name, pred_data in self.predictions.items():
            short_name = self._shorten_model_name(model_name)
            predicted = pred_data['predicted']
            actual = pred_data['actual']
            
            # Calculate various error metrics
            abs_error = np.abs(actual - predicted)
            rel_error = np.abs((actual - predicted) / actual) * 100
            squared_error = (actual - predicted) ** 2
            
            # Add to DataFrame
            base_df[f'Predicted_{model_name}'] = predicted
            base_df[f'Absolute_Error_{model_name}'] = abs_error
            base_df[f'Relative_Error_%_{model_name}'] = rel_error
            base_df[f'Squared_Error_{model_name}'] = squared_error
            
            # Add error direction
            base_df[f'Error_Direction_{model_name}'] = np.where(
                predicted > actual, 'Over-prediction', 'Under-prediction'
            )
        
        return base_df
    
    def get_prediction_stats(self) -> pd.DataFrame:
        """Get comprehensive prediction statistics for each model."""
        stats_data = []
        
        for model_name, pred_data in self.predictions.items():
            actual = pred_data['actual']
            predicted = pred_data['predicted']
            errors = np.abs(actual - predicted)
            rel_errors = np.abs((actual - predicted) / actual) * 100
            
            stats = {
                'Model': model_name,
                'Mean Error': np.mean(errors),
                'Median Error': np.median(errors),
                'Std Error': np.std(errors),
                'Min Error': np.min(errors),
                'Max Error': np.max(errors),
                'Mean Relative Error %': np.mean(rel_errors),
                'Median Relative Error %': np.median(rel_errors),
                '90th Percentile Error': np.percentile(errors, 90),
                '95th Percentile Error': np.percentile(errors, 95),
                'Over-predictions %': np.mean(predicted > actual) * 100,
                'Under-predictions %': np.mean(predicted < actual) * 100
            }
            
            stats_data.append(stats)
        
        return pd.DataFrame(stats_data).set_index('Model')
    
    #####################################################################

    def plot_metrics_comparison(self, metric: str = 'r2') -> go.Figure:
        """Create an enhanced bar plot comparing a specific metric across models."""
        metrics_df = self.get_metrics_comparison()
        
        if metric not in metrics_df.index:
            raise ValueError(f"Metric '{metric}' not found. Available metrics: {metrics_df.index.tolist()}")
        
        # Determine if lower is better for this metric
        lower_is_better = metric.lower() in ['mse', 'rmse', 'mae', 'mape']

        # Create mapping of full names to short names
        short_names = {name: self._shorten_model_name(name) for name in metrics_df.columns}
        
        # Sort values and get shortened names
        sorted_data = metrics_df.loc[metric].sort_values(
            ascending=metric.lower() in ['mse', 'rmse', 'mae', 'mape']
        )
        sorted_short_names = [short_names[name] for name in sorted_data.index]
        
        # Create color scale based on values
        colors = [
            'rgb(50,205,50)' if i == 0 else  # Best performer
            'rgb(255,165,0)' if i == len(sorted_data) - 1 else  # Worst performer
            'rgb(30,144,255)'  # Others
            for i in range(len(sorted_data))
        ]
        
        fig = go.Figure(data=[
            go.Bar(
                x=sorted_data.index,
                y=sorted_data.values,
                text=sorted_data.round(4),
                textposition='auto',
                marker_color=colors
            )
        ])
        
        # Add reference line for best score if appropriate
        if metric == 'r2':
            fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                         annotation_text="Perfect Score (R²=1)")
        
        fig.update_layout(
            title=f'{metric.upper()} Score Comparison<br><sup>{self.metric_descriptions.get(metric, "")}</sup>',
            xaxis_title='Models',
            yaxis_title=f'{metric.upper()} Score',
            template='plotly_white',
            showlegend=False,
            xaxis_tickangle=45  # Angle labels for better readability
        )
        
        return fig
    
    def plot_prediction_comparison(self) -> go.Figure:
        """Create an enhanced scatter plot comparing predictions."""
        if not self.predictions:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Predictions Comparison', 'Error Distribution'),
            vertical_spacing=0.2,
            row_heights=[0.7, 0.3]
        )

        # Get shortened names
        short_names = {name: self._shorten_model_name(name) 
                      for name in self.predictions.keys()}
        

        # Find global min and max
        all_actuals = np.concatenate([pred_data['actual'] for pred_data in self.predictions.values()])
        all_predictions = np.concatenate([pred_data['predicted'] for pred_data in self.predictions.values()])
        min_val = min(np.min(all_actuals), np.min(all_predictions))
        max_val = max(np.max(all_actuals), np.max(all_predictions))
        
        # Add perfect prediction line
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='black', dash='dash')
            ),
            row=1, col=1
        )
        
        # Add model predictions
        colors = px.colors.qualitative.Set3
        for i, (model_name, pred_data) in enumerate(self.predictions.items()):
            short_name = short_names[model_name]

            # Scatter plot
            fig.add_trace(
                go.Scatter(
                    x=pred_data['actual'],
                    y=pred_data['predicted'],
                    mode='markers',
                    name=short_name,
                    marker=dict(
                        size=8,
                        color=colors[i % len(colors)]
                    )
                ),
                row=1, col=1
            )
            
            # Error distribution
            errors = pred_data['predicted'] - pred_data['actual']
            fig.add_trace(
                go.Histogram(
                    x=errors,
                    name=f"{short_name} Errors",
                    nbinsx=30,
                    opacity=0.7,
                    marker_color=colors[i % len(colors)]
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            height=800,
            title='Prediction Analysis',
            template='plotly_white'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Actual Values", row=1, col=1)
        fig.update_yaxes(title_text="Predicted Values", row=1, col=1)
        fig.update_xaxes(title_text="Prediction Error", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        
        return fig
    
    def plot_feature_importance_comparison(self) -> Optional[go.Figure]:
        """Create a heatmap comparing feature importance across models."""
        importance_data = {}
        for model_name, result in self.results.items():
            if result['feature_importance'] is not None:
                importance_data[self._shorten_model_name(model_name)] = result['feature_importance']
        
        if not importance_data:
            return None
        
        # Create DataFrame and sort features by average importance
        importance_df = pd.DataFrame(importance_data)
        avg_importance = importance_df.mean(axis=1)
        importance_df = importance_df.loc[avg_importance.sort_values(ascending=False).index]
        
        fig = go.Figure(data=go.Heatmap(
            z=importance_df.values,
            x=importance_df.columns,
            y=importance_df.index,
            colorscale='RdBu',
            text=np.round(importance_df.values, 4),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Feature Importance Comparison<br><sup>Darker colors indicate higher importance</sup>',
            xaxis_title='Models',
            yaxis_title='Features',
            template='plotly_white',
            height=max(400, len(importance_df) * 30),  # Dynamic height based on number of features
            xaxis_tickangle=45  # Angle model names for better readability
        )
        
        return fig
    ##################################################################

    def plot_error_analysis(self) -> go.Figure:
        """Create a comprehensive error analysis visualization."""
        if not self.predictions:
            return go.Figure()
        
        stats_df = self.get_prediction_stats()
        
        # Create mapping of full names to short names
        short_names = {model_name: self._shorten_model_name(model_name) 
                      for model_name in stats_df.index}
        
        # Create new index with shortened names
        stats_df.index = [short_names[name] for name in stats_df.index]
        
        # Create subplots for different error metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Mean Error by Model',
                'Error Distribution',
                'Error Ranges',
                'Relative Error %'
            ),
            vertical_spacing=0.2,
            horizontal_spacing=0.15
        )
        
        colors = px.colors.qualitative.Set3
        
        # Mean Error Bar Plot
        fig.add_trace(
            go.Bar(
                x=stats_df.index,
                y=stats_df['Mean Error'],
                marker_color=colors,
                name='Mean Error'
            ),
            row=1, col=1
        )
        
        # Error Distribution (Box Plot)
        for i, (model_name, short_name) in enumerate(short_names.items()):
            pred_data = self.predictions[model_name]
            errors = np.abs(pred_data['predicted'] - pred_data['actual'])
            
            fig.add_trace(
                go.Box(
                    y=errors,
                    name=short_name,
                    marker_color=colors[i % len(colors)]
                ),
                row=1, col=2
            )
        
        # Error Ranges
        for i, short_name in enumerate(stats_df.index):
            fig.add_trace(
                go.Scatter(
                    x=[stats_df.loc[short_name, 'Min Error'],
                       stats_df.loc[short_name, 'Mean Error'],
                       stats_df.loc[short_name, 'Max Error']],
                    y=[short_name] * 3,
                    mode='lines+markers',
                    name=short_name,
                    marker_color=colors[i % len(colors)]
                ),
                row=2, col=1
            )
        
        # Relative Error
        fig.add_trace(
            go.Bar(
                x=stats_df.index,
                y=stats_df['Mean Relative Error %'],
                marker_color=colors,
                name='Mean Relative Error %'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title='Comprehensive Error Analysis',
            showlegend=False,
            template='plotly_white'
        )
        
        # Update layout for better readability
        fig.update_xaxes(tickangle=45)  # Angle the labels for better fit
        fig.update_layout(margin=dict(l=50, r=50, t=100, b=50))  # Adjust margins
        
        return fig

    ##################################################################

    def generate_comparison_insights(self) -> Dict[str, Any]:
        """Generate comprehensive insights from the model comparison."""
        insights = {
            'overall_best': {},
            'metric_analysis': {},
            'prediction_patterns': {},
            'feature_importance': {},
            'recommendations': []
        }
        
        # Overall best model analysis
        metrics_df = self.get_metrics_comparison()
        model_scores = {}
        
        for model in metrics_df.columns:
            score = 0
            for metric in metrics_df.index:
                if metric == 'r2':
                    score += (metrics_df.loc[metric, model] == metrics_df.loc[metric].max()) * 2
                elif metric in ['mse', 'rmse', 'mae', 'mape']:
                    score += (metrics_df.loc[metric, model] == metrics_df.loc[metric].min())
            model_scores[model] = score
        
        best_model = max(model_scores.items(), key=lambda x: x[1])[0]
        insights['overall_best'] = {
            'model': best_model,
            'strengths': [
                metric for metric in metrics_df.index
                if (metric == 'r2' and metrics_df.loc[metric, best_model] == metrics_df.loc[metric].max()) or
                (metric in ['mse', 'rmse', 'mae', 'mape'] and metrics_df.loc[metric, best_model] == metrics_df.loc[metric].min())
            ]
        }
        
        # Metric analysis
        for metric in metrics_df.index:
            best_model = self.get_best_model(metric)
            worst_model = self.get_worst_model(metric)
            improvement = abs(
                metrics_df.loc[metric, best_model] - metrics_df.loc[metric, worst_model]
            ) / abs(metrics_df.loc[metric, worst_model]) * 100
            
            insights['metric_analysis'][metric] = {
                'best_model': best_model,
                'worst_model': worst_model,
                'improvement_percentage': improvement,
                'significance': 'High' if improvement > 20 else 'Medium' if improvement > 10 else 'Low'
            }
        
        # Prediction pattern analysis
        prediction_stats = self.get_prediction_stats()
        for model in prediction_stats.index:
            insights['prediction_patterns'][model] = {
                'bias_tendency': 'Over-prediction' if prediction_stats.loc[model, 'Over-predictions %'] > 55 else
                                'Under-prediction' if prediction_stats.loc[model, 'Under-predictions %'] > 55 else
                                'Balanced',
                'error_distribution': 'Consistent' if prediction_stats.loc[model, 'Std Error'] < 
                                    prediction_stats['Std Error'].mean() else 'Variable',
                'reliability': 'High' if prediction_stats.loc[model, 'Mean Relative Error %'] < 
                              prediction_stats['Mean Relative Error %'].mean() else 'Medium'
            }
        
        # Feature importance analysis
        if any(result['feature_importance'] is not None for result in self.results.values()):
            common_features = self.get_common_important_features()
            insights['feature_importance'] = {
                'common_important_features': common_features,
                'consistency': 'High' if len(common_features) >= 3 else 'Medium' if len(common_features) >= 2 else 'Low'
            }
        
        # Generate recommendations
        insights['recommendations'] = self._generate_recommendations(insights)
        
        return insights
    
    def get_stable_features(self) -> Dict[str, float]:
        """
        Get features that have stable importance across models.
        Returns a dictionary of features and their stability scores.
        """
        importance_data = {}
        for model_name, result in self.results.items():
            if result['feature_importance'] is not None:
                importance_data[model_name] = result['feature_importance']
        
        if not importance_data:
            return {}
        
        # Create DataFrame of feature importance
        importance_df = pd.DataFrame(importance_data)
        
        # Calculate stability score for each feature
        stability_scores = {}
        for feature in importance_df.index:
            values = importance_df.loc[feature]
            # Calculate coefficient of variation (lower means more stable)
            cv = values.std() / values.mean() if values.mean() != 0 else float('inf')
            # Convert to stability score (1 - normalized cv), higher is better
            stability_score = 1 - (cv / (1 + cv))
            stability_scores[feature] = stability_score
        
        # Sort by stability score and return top features
        return dict(sorted(stability_scores.items(), key=lambda x: x[1], reverse=True))

    def get_common_important_features(self, top_n: int = 3) -> Dict[str, List[str]]:
        """Get features that are consistently important across models."""
        important_features = {}
        
        for model_name, result in self.results.items():
            if result['feature_importance'] is not None:
                # Get top N features for this model
                top_features = sorted(
                    result['feature_importance'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:top_n]
                
                for feature, _ in top_features:
                    if feature not in important_features:
                        important_features[feature] = []
                    important_features[feature].append(model_name)
        
        return dict(sorted(
            important_features.items(),
            key=lambda x: len(x[1]),
            reverse=True
        ))

    def get_best_model(self, metric: str = 'r2') -> str:
        """Get the best performing model for a specific metric."""
        metrics_df = self.get_metrics_comparison()
        if metric not in metrics_df.index:
            raise ValueError(f"Metric '{metric}' not found. Available metrics: {metrics_df.index.tolist()}")
        
        # For these metrics, lower is better
        lower_is_better = metric.lower() in ['mse', 'rmse', 'mae', 'mape']
        
        if lower_is_better:
            return metrics_df.loc[metric].idxmin()
        return metrics_df.loc[metric].idxmax()    
    
    def get_worst_model(self, metric: str = 'r2') -> str:
        """Get the name of the worst performing model based on a specific metric."""
        metrics_df = self.get_metrics_comparison()
        if metric not in metrics_df.index:
            raise ValueError(f"Metric '{metric}' not found. Available metrics: {metrics_df.index.tolist()}")
            
        # Handle metrics where lower is better
        lower_is_better = metric.lower() in ['mse', 'rmse', 'mae']
        if lower_is_better:
            return metrics_df.loc[metric].idxmax()
        return metrics_df.loc[metric].idxmin()
    
    def _generate_recommendations(self, insights: Dict) -> List[str]:
        """Generate specific recommendations based on the analysis."""
        recommendations = []
        
        # Best model recommendation
        recommendations.append(
            f"The overall best performing model is {insights['overall_best']['model']}, "
            f"showing particular strength in {', '.join(insights['overall_best']['strengths'])}."
        )
        
        # Specific use-case recommendations
        for model, patterns in insights['prediction_patterns'].items():
            if patterns['bias_tendency'] == 'Balanced' and patterns['reliability'] == 'High':
                recommendations.append(
                    f"{model} shows balanced predictions with high reliability, "
                    "making it suitable for general-purpose use."
                )
            elif patterns['bias_tendency'] == 'Over-prediction':
                recommendations.append(
                    f"{model} tends to over-predict, consider using it when "
                    "conservative estimates are preferred."
                )
            elif patterns['bias_tendency'] == 'Under-prediction':
                recommendations.append(
                    f"{model} tends to under-predict, consider using it when "
                    "aggressive estimates are acceptable."
                )
        
        # Feature importance based recommendations
        if 'feature_importance' in insights and insights['feature_importance']['consistency'] == 'High':
            top_features = list(insights['feature_importance']['common_important_features'].keys())[:3]
            recommendations.append(
                f"All models consistently identify {', '.join(top_features)} as the most "
                "important features. Consider focusing on these for future modeling."
            )
        
        return recommendations
    
    def get_model_rankings(self, metric: str = 'r2') -> Dict[str, float]:
        """Get models ranked by a specific metric."""
        metrics_df = self.get_metrics_comparison()
        if metric not in metrics_df.index:
            raise ValueError(f"Metric '{metric}' not found. Available metrics: {metrics_df.index.tolist()}")
        
        # For these metrics, lower is better
        lower_is_better = metric.lower() in ['mse', 'rmse', 'mae', 'mape']
        
        sorted_series = metrics_df.loc[metric].sort_values(ascending=lower_is_better)
        return sorted_series.to_dict()    
    
    def export_comparison_report(self) -> Dict[str, Any]:
        """Export a comprehensive comparison report with insights."""
        metrics_df = self.get_metrics_comparison()
        insights = self.generate_comparison_insights()
        
        return {
            'metrics_comparison': metrics_df,
            'prediction_comparison': self.get_prediction_comparison(),
            'prediction_stats': self.get_prediction_stats(),
            'model_rankings': {
                metric: self.get_model_rankings(metric).to_dict()
                for metric in metrics_df.index
            },
            'best_models': {
                metric: self.get_best_model(metric)
                for metric in metrics_df.index
            },
            'insights': insights
        }