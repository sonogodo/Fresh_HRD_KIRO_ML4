"""
Monitoring module for Decision ML pipeline.
Provides logging, metrics tracking, and model drift detection.
"""

import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import os
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy import stats
import warnings

class DecisionMLMonitor:
    """
    Monitoring system for Decision ML pipeline.
    Tracks model performance, data drift, and system health.
    """
    
    def __init__(self, log_dir: str = "decision_ml/logs/"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Metrics storage
        self.metrics_history = []
        self.drift_alerts = []
        self.performance_baseline = None
        
        # Drift detection parameters
        self.drift_threshold = 0.05  # p-value threshold for statistical tests
        self.performance_threshold = 0.1  # 10% performance degradation threshold
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('decision_ml_monitor')
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = os.path.join(self.log_dir, f"decision_ml_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_prediction_request(self, request_data: Dict[str, Any], 
                             predictions: List[Dict], response_time: float) -> None:
        """Log prediction request details."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'prediction_request',
            'request_size': len(request_data.get('candidate_data', [])) if request_data.get('candidate_data') else 0,
            'predictions_count': len(predictions),
            'response_time_ms': response_time * 1000,
            'top_k': request_data.get('top_k', 3)
        }
        
        self.logger.info(f"Prediction request: {json.dumps(log_entry)}")
        
        # Store metrics
        self.metrics_history.append(log_entry)
    
    def log_training_event(self, training_results: Dict[str, Any]) -> None:
        """Log model training event."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'model_training',
            'best_model': training_results.get('best_model_name'),
            'best_score': training_results.get('best_score'),
            'models_trained': len(training_results.get('all_results', {}))
        }
        
        self.logger.info(f"Model training completed: {json.dumps(log_entry)}")
        
        # Update performance baseline
        if training_results.get('best_score'):
            self.performance_baseline = training_results['best_score']
    
    def log_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None) -> None:
        """Log error events."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'error',
            'error_type': error_type,
            'error_message': error_message,
            'context': context or {}
        }
        
        self.logger.error(f"Error occurred: {json.dumps(log_entry)}")
    
    def detect_data_drift(self, reference_data: pd.DataFrame, 
                         current_data: pd.DataFrame, 
                         feature_columns: List[str]) -> Dict[str, Any]:
        """Detect data drift using statistical tests."""
        self.logger.info("Starting data drift detection...")
        
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': False,
            'feature_drift_scores': {},
            'significant_drifts': [],
            'overall_drift_score': 0.0
        }
        
        drift_scores = []
        
        for feature in feature_columns:
            if feature in reference_data.columns and feature in current_data.columns:
                try:
                    # Use Kolmogorov-Smirnov test for continuous features
                    ref_values = reference_data[feature].dropna()
                    curr_values = current_data[feature].dropna()
                    
                    if len(ref_values) > 0 and len(curr_values) > 0:
                        ks_statistic, p_value = stats.ks_2samp(ref_values, curr_values)
                        
                        drift_results['feature_drift_scores'][feature] = {
                            'ks_statistic': float(ks_statistic),
                            'p_value': float(p_value),
                            'drift_detected': p_value < self.drift_threshold
                        }
                        
                        drift_scores.append(ks_statistic)
                        
                        if p_value < self.drift_threshold:
                            drift_results['significant_drifts'].append(feature)
                            self.logger.warning(f"Data drift detected in feature '{feature}': p-value = {p_value:.6f}")
                
                except Exception as e:
                    self.logger.error(f"Error detecting drift for feature '{feature}': {str(e)}")
        
        # Calculate overall drift score
        if drift_scores:
            drift_results['overall_drift_score'] = float(np.mean(drift_scores))
            drift_results['drift_detected'] = len(drift_results['significant_drifts']) > 0
        
        # Log drift detection results
        if drift_results['drift_detected']:
            self.drift_alerts.append(drift_results)
            self.logger.warning(f"Data drift detected: {json.dumps(drift_results)}")
        else:
            self.logger.info("No significant data drift detected")
        
        return drift_results
    
    def monitor_model_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                y_pred_proba: np.ndarray = None) -> Dict[str, Any]:
        """Monitor model performance and detect degradation."""
        self.logger.info("Monitoring model performance...")
        
        # Calculate current performance metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        performance_metrics = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': float(accuracy),
            'sample_size': len(y_true),
            'positive_rate': float(np.mean(y_true))
        }
        
        if y_pred_proba is not None:
            try:
                roc_auc = roc_auc_score(y_true, y_pred_proba)
                performance_metrics['roc_auc'] = float(roc_auc)
            except ValueError:
                # Handle case where only one class is present
                performance_metrics['roc_auc'] = None
        
        # Check for performance degradation
        performance_alert = False
        if self.performance_baseline is not None:
            current_score = performance_metrics.get('roc_auc', accuracy)
            degradation = (self.performance_baseline - current_score) / self.performance_baseline
            
            if degradation > self.performance_threshold:
                performance_alert = True
                self.logger.warning(
                    f"Model performance degradation detected: "
                    f"baseline={self.performance_baseline:.4f}, "
                    f"current={current_score:.4f}, "
                    f"degradation={degradation:.2%}"
                )
        
        performance_metrics['performance_alert'] = performance_alert
        performance_metrics['baseline_score'] = self.performance_baseline
        
        # Log performance metrics
        self.logger.info(f"Model performance: {json.dumps(performance_metrics)}")
        
        return performance_metrics
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        now = datetime.now()
        
        # Recent metrics (last 24 hours)
        recent_metrics = [
            m for m in self.metrics_history 
            if datetime.fromisoformat(m['timestamp']) > now - timedelta(days=1)
        ]
        
        # Recent drift alerts (last 7 days)
        recent_drift_alerts = [
            alert for alert in self.drift_alerts
            if datetime.fromisoformat(alert['timestamp']) > now - timedelta(days=7)
        ]
        
        health_status = {
            'timestamp': now.isoformat(),
            'overall_status': 'healthy',
            'metrics': {
                'total_predictions_24h': len(recent_metrics),
                'avg_response_time_ms': np.mean([m.get('response_time_ms', 0) for m in recent_metrics]) if recent_metrics else 0,
                'drift_alerts_7d': len(recent_drift_alerts),
                'performance_baseline': self.performance_baseline
            },
            'alerts': {
                'active_drift_alerts': len(recent_drift_alerts),
                'recent_errors': self._count_recent_errors()
            }
        }
        
        # Determine overall health status
        if len(recent_drift_alerts) > 0 or health_status['alerts']['recent_errors'] > 10:
            health_status['overall_status'] = 'warning'
        
        if health_status['alerts']['recent_errors'] > 50:
            health_status['overall_status'] = 'critical'
        
        return health_status
    
    def _count_recent_errors(self) -> int:
        """Count errors in the last 24 hours from log files."""
        try:
            log_file = os.path.join(self.log_dir, f"decision_ml_{datetime.now().strftime('%Y%m%d')}.log")
            if not os.path.exists(log_file):
                return 0
            
            error_count = 0
            cutoff_time = datetime.now() - timedelta(days=1)
            
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if 'ERROR' in line:
                        try:
                            # Extract timestamp from log line
                            timestamp_str = line.split(' - ')[0]
                            log_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                            if log_time > cutoff_time:
                                error_count += 1
                        except:
                            continue
            
            return error_count
        except Exception:
            return 0
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        self.logger.info("Generating monitoring report...")
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'system_health': self.get_system_health(),
            'recent_metrics': self.metrics_history[-100:],  # Last 100 requests
            'drift_alerts': self.drift_alerts[-10:],  # Last 10 drift alerts
            'summary': {
                'total_requests': len(self.metrics_history),
                'total_drift_alerts': len(self.drift_alerts),
                'monitoring_period_days': (
                    (datetime.now() - datetime.fromisoformat(self.metrics_history[0]['timestamp'])).days
                    if self.metrics_history else 0
                )
            }
        }
        
        # Save report to file
        report_file = os.path.join(
            self.log_dir, 
            f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Monitoring report saved to: {report_file}")
        
        return report
    
    def clear_old_logs(self, days_to_keep: int = 30) -> None:
        """Clear old log files to manage disk space."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for filename in os.listdir(self.log_dir):
            if filename.startswith('decision_ml_') and filename.endswith('.log'):
                try:
                    # Extract date from filename
                    date_str = filename.replace('decision_ml_', '').replace('.log', '')
                    file_date = datetime.strptime(date_str, '%Y%m%d')
                    
                    if file_date < cutoff_date:
                        file_path = os.path.join(self.log_dir, filename)
                        os.remove(file_path)
                        self.logger.info(f"Removed old log file: {filename}")
                
                except Exception as e:
                    self.logger.error(f"Error removing old log file {filename}: {str(e)}")

# Global monitor instance
monitor = DecisionMLMonitor()