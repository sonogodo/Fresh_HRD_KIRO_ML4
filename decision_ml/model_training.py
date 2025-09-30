"""
Model training module for Decision recruitment ML pipeline.
Implements multiple ML models for job-candidate matching.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class DecisionModelTrainer:
    """
    Model trainer for Decision recruitment matching system.
    Supports multiple ML algorithms and hyperparameter tuning.
    """
    
    def __init__(self, model_save_path: str = "decision_ml/models/"):
        self.model_save_path = model_save_path
        self.models = {}
        self.scalers = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        
        # Ensure model directory exists
        os.makedirs(model_save_path, exist_ok=True)
        
        # Define model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'svm': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            }
        }
    
    def prepare_training_data(self, matching_df: pd.DataFrame, feature_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from matching dataframe."""
        logger.info("Preparing training data...")
        
        # Extract features and target
        X = matching_df[feature_columns].fillna(0)
        y = matching_df['is_good_match']
        
        logger.info(f"Training data shape: {X.shape}")
        logger.info(f"Positive samples: {y.sum()}, Negative samples: {len(y) - y.sum()}")
        
        return X.values, y.values
    
    def train_single_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray, 
                          X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train a single model with hyperparameter tuning."""
        logger.info(f"Training {model_name}...")
        
        config = self.model_configs[model_name]
        model = config['model']
        param_grid = config['params']
        
        # Scale features for models that need it
        if model_name in ['logistic_regression', 'svm']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            self.scalers[model_name] = scaler
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
        
        # Hyperparameter tuning with cross-validation
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='roc_auc', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = best_model.predict(X_val_scaled)
        y_pred_proba = best_model.predict_proba(X_val_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        
        # Cross-validation score
        cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        
        results = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'classification_report': classification_report(y_val, y_pred)
        }
        
        self.models[model_name] = results
        
        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
        logger.info(f"{model_name} - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, Any]:
        """Train all configured models and compare performance."""
        logger.info("Starting model training pipeline...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train all models
        for model_name in self.model_configs.keys():
            try:
                self.train_single_model(model_name, X_train, y_train, X_val, y_val)
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        # Select best model based on ROC-AUC
        best_score = 0
        for model_name, results in self.models.items():
            if results['roc_auc'] > best_score:
                best_score = results['roc_auc']
                self.best_model = results['model']
                self.best_model_name = model_name
        
        logger.info(f"Best model: {self.best_model_name} with ROC-AUC: {best_score:.4f}")
        
        # Extract feature importance if available
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            self.feature_importance = np.abs(self.best_model.coef_[0])
        
        return {
            'best_model_name': self.best_model_name,
            'best_score': best_score,
            'all_results': self.models,
            'validation_data': (X_val, y_val)
        }
    
    def save_models(self) -> None:
        """Save all trained models and scalers."""
        logger.info("Saving models...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save best model
        if self.best_model is not None:
            model_path = os.path.join(self.model_save_path, f"best_model_{timestamp}.joblib")
            joblib.dump(self.best_model, model_path)
            logger.info(f"Best model saved to: {model_path}")
            
            # Save model metadata
            metadata = {
                'model_name': self.best_model_name,
                'model_type': type(self.best_model).__name__,
                'training_timestamp': timestamp,
                'performance_metrics': self.models[self.best_model_name]
            }
            
            metadata_path = os.path.join(self.model_save_path, f"model_metadata_{timestamp}.joblib")
            joblib.dump(metadata, metadata_path)
        
        # Save scalers
        for model_name, scaler in self.scalers.items():
            scaler_path = os.path.join(self.model_save_path, f"scaler_{model_name}_{timestamp}.joblib")
            joblib.dump(scaler, scaler_path)
        
        # Save all models
        all_models_path = os.path.join(self.model_save_path, f"all_models_{timestamp}.joblib")
        joblib.dump(self.models, all_models_path)
    
    def load_model(self, model_path: str) -> Any:
        """Load a saved model."""
        return joblib.load(model_path)
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, feature_names: List[str] = None) -> Dict[str, Any]:
        """Evaluate the best model on test data."""
        if self.best_model is None:
            raise ValueError("No trained model available. Train a model first.")
        
        logger.info("Evaluating best model...")
        
        # Scale test data if needed
        if self.best_model_name in self.scalers:
            X_test_scaled = self.scalers[self.best_model_name].transform(X_test)
        else:
            X_test_scaled = X_test
        
        # Make predictions
        y_pred = self.best_model.predict(X_test_scaled)
        y_pred_proba = self.best_model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        evaluation_results = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test ROC-AUC: {roc_auc:.4f}")
        
        return evaluation_results
    
    def predict_match_probability(self, features: np.ndarray) -> np.ndarray:
        """Predict match probability for new job-candidate pairs."""
        if self.best_model is None:
            raise ValueError("No trained model available. Train a model first.")
        
        # Scale features if needed
        if self.best_model_name in self.scalers:
            features_scaled = self.scalers[self.best_model_name].transform(features)
        else:
            features_scaled = features
        
        return self.best_model.predict_proba(features_scaled)[:, 1]
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of all trained models."""
        if not self.models:
            return {"message": "No models trained yet"}
        
        summary = {
            'best_model': self.best_model_name,
            'models_trained': len(self.models),
            'model_performance': {}
        }
        
        for model_name, results in self.models.items():
            summary['model_performance'][model_name] = {
                'accuracy': results['accuracy'],
                'roc_auc': results['roc_auc'],
                'cv_mean': results['cv_mean'],
                'cv_std': results['cv_std']
            }
        
        return summary