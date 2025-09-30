"""
Main ML pipeline for Decision recruitment system.
Orchestrates data preprocessing, feature engineering, model training, and prediction.
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Any, Tuple
import os
from datetime import datetime

from decision_ml.data_preprocessing import DecisionDataPreprocessor
from decision_ml.feature_engineering import DecisionFeatureEngineer
from decision_ml.model_training import DecisionModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DecisionMLPipeline:
    """
    Complete ML pipeline for Decision recruitment matching system.
    """
    
    def __init__(self, model_save_path: str = "decision_ml/models/"):
        self.preprocessor = DecisionDataPreprocessor()
        self.feature_engineer = DecisionFeatureEngineer()
        self.model_trainer = DecisionModelTrainer(model_save_path)
        
        self.jobs_df = None
        self.candidates_df = None
        self.matching_df = None
        self.feature_columns = None
        
        # Pipeline state
        self.is_trained = False
        self.training_timestamp = None
        
    def load_data(self, jobs_path: str, candidates_path: str) -> None:
        """Load and preprocess data from JSON files."""
        logger.info("Loading data...")
        
        # Load jobs data
        with open(jobs_path, 'r', encoding='utf-8') as f:
            jobs_data = json.load(f)
        
        # Load candidates data
        with open(candidates_path, 'r', encoding='utf-8') as f:
            candidates_data = json.load(f)
        
        # Preprocess data
        self.jobs_df = self.preprocessor.preprocess_jobs_data(jobs_data)
        self.candidates_df = self.preprocessor.preprocess_candidates_data(candidates_data)
        
        logger.info(f"Loaded {len(self.jobs_df)} jobs and {len(self.candidates_df)} candidates")
    
    def create_features(self, threshold: float = 70.0) -> None:
        """Create features for ML training."""
        if self.jobs_df is None or self.candidates_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info("Creating features...")
        
        # Create matching features
        self.matching_df = self.feature_engineer.create_matching_features(
            self.jobs_df, self.candidates_df
        )
        
        # Add ranking features
        self.matching_df = self.feature_engineer.create_candidate_ranking_features(
            self.matching_df
        )
        
        # Create binary labels
        self.matching_df = self.feature_engineer.create_binary_labels(
            self.matching_df, threshold
        )
        
        # Get feature columns
        self.feature_columns = self.feature_engineer.get_feature_columns()
        
        logger.info(f"Created {len(self.matching_df)} job-candidate pairs with {len(self.feature_columns)} features")
    
    def train_models(self, test_size: float = 0.2) -> Dict[str, Any]:
        """Train ML models."""
        if self.matching_df is None:
            raise ValueError("Features not created. Call create_features() first.")
        
        logger.info("Training models...")
        
        # Prepare training data
        X, y = self.model_trainer.prepare_training_data(self.matching_df, self.feature_columns)
        
        # Train all models
        training_results = self.model_trainer.train_all_models(X, y, test_size)
        
        # Save models
        self.model_trainer.save_models()
        
        # Update pipeline state
        self.is_trained = True
        self.training_timestamp = datetime.now()
        
        logger.info("Model training completed successfully")
        
        return training_results
    
    def predict_matches(self, job_data: Dict = None, candidate_data: List[Dict] = None, 
                       top_k: int = 3) -> List[Dict[str, Any]]:
        """Predict top matches for given job(s) and candidates."""
        if not self.is_trained:
            raise ValueError("Pipeline not trained. Call train_models() first.")
        
        logger.info("Predicting matches...")
        
        # Use existing data if not provided
        if job_data is None and candidate_data is None:
            jobs_to_process = self.jobs_df
            candidates_to_process = self.candidates_df
        else:
            # Process new data
            if job_data:
                jobs_to_process = self.preprocessor.preprocess_jobs_data({1: job_data})
            else:
                jobs_to_process = self.jobs_df
            
            if candidate_data:
                candidates_to_process = self.preprocessor.preprocess_candidates_data(candidate_data)
            else:
                candidates_to_process = self.candidates_df
        
        # Create features for prediction
        prediction_df = self.feature_engineer.create_matching_features(
            jobs_to_process, candidates_to_process
        )
        
        prediction_df = self.feature_engineer.create_candidate_ranking_features(
            prediction_df
        )
        
        # Prepare features
        X_pred = prediction_df[self.feature_columns].fillna(0).values
        
        # Predict probabilities
        match_probabilities = self.model_trainer.predict_match_probability(X_pred)
        prediction_df['match_probability'] = match_probabilities
        
        # Generate top matches for each job
        results = []
        for job_id in prediction_df['job_id'].unique():
            job_matches = prediction_df[prediction_df['job_id'] == job_id].copy()
            job_matches = job_matches.sort_values('match_probability', ascending=False)
            
            top_matches = []
            for _, match in job_matches.head(top_k).iterrows():
                top_matches.append({
                    'candidate_id': match['candidate_id'],
                    'match_probability': float(match['match_probability']),
                    'overall_score': float(match['overall_score']),
                    'skill_match_score': float(match['skill_match_score']),
                    'experience_compatibility': float(match['experience_compatibility']),
                    'education_compatibility': float(match['education_compatibility']),
                    'language_compatibility': float(match['language_compatibility']),
                    'text_similarity': float(match['text_similarity'])
                })
            
            results.append({
                'job_id': job_id,
                'top_matches': top_matches
            })
        
        return results
    
    def evaluate_pipeline(self, test_jobs_path: str = None, test_candidates_path: str = None) -> Dict[str, Any]:
        """Evaluate the trained pipeline on test data."""
        if not self.is_trained:
            raise ValueError("Pipeline not trained. Call train_models() first.")
        
        if test_jobs_path and test_candidates_path:
            # Load test data
            with open(test_jobs_path, 'r', encoding='utf-8') as f:
                test_jobs_data = json.load(f)
            
            with open(test_candidates_path, 'r', encoding='utf-8') as f:
                test_candidates_data = json.load(f)
            
            # Preprocess test data
            test_jobs_df = self.preprocessor.preprocess_jobs_data(test_jobs_data)
            test_candidates_df = self.preprocessor.preprocess_candidates_data(test_candidates_data)
            
            # Create test features
            test_matching_df = self.feature_engineer.create_matching_features(
                test_jobs_df, test_candidates_df
            )
            test_matching_df = self.feature_engineer.create_candidate_ranking_features(
                test_matching_df
            )
            test_matching_df = self.feature_engineer.create_binary_labels(test_matching_df)
            
            # Prepare test data
            X_test = test_matching_df[self.feature_columns].fillna(0).values
            y_test = test_matching_df['is_good_match'].values
            
        else:
            # Use validation data from training
            _, (X_test, y_test) = self.model_trainer.train_all_models.__wrapped__(
                self.model_trainer, 
                self.matching_df[self.feature_columns].fillna(0).values,
                self.matching_df['is_good_match'].values
            )
        
        # Evaluate model
        evaluation_results = self.model_trainer.evaluate_model(X_test, y_test, self.feature_columns)
        
        return evaluation_results
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline summary."""
        summary = {
            'pipeline_status': 'trained' if self.is_trained else 'not_trained',
            'training_timestamp': self.training_timestamp.isoformat() if self.training_timestamp else None,
            'data_summary': {
                'jobs_count': len(self.jobs_df) if self.jobs_df is not None else 0,
                'candidates_count': len(self.candidates_df) if self.candidates_df is not None else 0,
                'job_candidate_pairs': len(self.matching_df) if self.matching_df is not None else 0,
                'feature_count': len(self.feature_columns) if self.feature_columns else 0
            }
        }
        
        if self.is_trained:
            summary['model_summary'] = self.model_trainer.get_model_summary()
        
        return summary
    
    def save_pipeline_state(self, save_path: str = "decision_ml/pipeline_state.json") -> None:
        """Save pipeline state for later loading."""
        state = {
            'is_trained': self.is_trained,
            'training_timestamp': self.training_timestamp.isoformat() if self.training_timestamp else None,
            'feature_columns': self.feature_columns,
            'pipeline_summary': self.get_pipeline_summary()
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Pipeline state saved to: {save_path}")

def run_complete_pipeline(jobs_path: str, candidates_path: str, 
                         model_save_path: str = "decision_ml/models/") -> DecisionMLPipeline:
    """Run the complete ML pipeline from start to finish."""
    logger.info("Starting complete Decision ML pipeline...")
    
    # Initialize pipeline
    pipeline = DecisionMLPipeline(model_save_path)
    
    # Load data
    pipeline.load_data(jobs_path, candidates_path)
    
    # Create features
    pipeline.create_features()
    
    # Train models
    training_results = pipeline.train_models()
    
    # Save pipeline state
    pipeline.save_pipeline_state()
    
    logger.info("Complete pipeline execution finished successfully")
    logger.info(f"Best model: {training_results['best_model_name']} with ROC-AUC: {training_results['best_score']:.4f}")
    
    return pipeline