#!/usr/bin/env python3
"""
Training script for Decision ML model.
Run this script to train the ML model with the Decision dataset.
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from decision_ml.pipeline import run_complete_pipeline
from decision_ml.monitoring import monitor

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Decision ML model')
    parser.add_argument(
        '--jobs-path', 
        default='JSONs_DECISION/vagas_padrao.json',
        help='Path to jobs JSON file'
    )
    parser.add_argument(
        '--candidates-path',
        default='JSONs_DECISION/candidates.json', 
        help='Path to candidates JSON file'
    )
    parser.add_argument(
        '--model-save-path',
        default='decision_ml/models/',
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Check if input files exist
    if not os.path.exists(args.jobs_path):
        logger.error(f"Jobs file not found: {args.jobs_path}")
        sys.exit(1)
    
    if not os.path.exists(args.candidates_path):
        logger.error(f"Candidates file not found: {args.candidates_path}")
        sys.exit(1)
    
    try:
        logger.info("Starting Decision ML model training...")
        logger.info(f"Jobs file: {args.jobs_path}")
        logger.info(f"Candidates file: {args.candidates_path}")
        logger.info(f"Model save path: {args.model_save_path}")
        
        # Run the complete pipeline
        pipeline = run_complete_pipeline(
            jobs_path=args.jobs_path,
            candidates_path=args.candidates_path,
            model_save_path=args.model_save_path
        )
        
        # Log training completion
        training_results = {
            'best_model_name': pipeline.model_trainer.best_model_name,
            'best_score': pipeline.model_trainer.models[pipeline.model_trainer.best_model_name]['roc_auc'],
            'all_results': pipeline.model_trainer.models
        }
        
        monitor.log_training_event(training_results)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Best Model: {training_results['best_model_name']}")
        logger.info(f"Best ROC-AUC Score: {training_results['best_score']:.4f}")
        logger.info(f"Total Jobs Processed: {len(pipeline.jobs_df)}")
        logger.info(f"Total Candidates Processed: {len(pipeline.candidates_df)}")
        logger.info(f"Total Job-Candidate Pairs: {len(pipeline.matching_df)}")
        logger.info(f"Models Saved to: {args.model_save_path}")
        logger.info("=" * 60)
        
        # Print model performance summary
        logger.info("\nModel Performance Summary:")
        for model_name, results in pipeline.model_trainer.models.items():
            logger.info(f"  {model_name}:")
            logger.info(f"    Accuracy: {results['accuracy']:.4f}")
            logger.info(f"    ROC-AUC: {results['roc_auc']:.4f}")
            logger.info(f"    CV Score: {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")
        
        # Test prediction
        logger.info("\nTesting prediction functionality...")
        test_predictions = pipeline.predict_matches(top_k=3)
        logger.info(f"Generated predictions for {len(test_predictions)} jobs")
        
        if test_predictions:
            logger.info(f"Sample prediction for job {test_predictions[0]['job_id']}:")
            for i, match in enumerate(test_predictions[0]['top_matches'][:3]):
                logger.info(f"  {i+1}. {match['candidate_id']} (score: {match['match_probability']:.3f})")
        
        logger.info("\nTraining completed successfully! 🎉")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        monitor.log_error('training_error', str(e), {'args': vars(args)})
        sys.exit(1)

if __name__ == '__main__':
    main()