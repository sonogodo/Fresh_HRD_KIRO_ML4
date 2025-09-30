"""
Unit tests for Decision ML pipeline components.
"""

import unittest
import pandas as pd
import numpy as np
import json
import tempfile
import os
from unittest.mock import patch, MagicMock

from decision_ml.data_preprocessing import DecisionDataPreprocessor
from decision_ml.feature_engineering import DecisionFeatureEngineer
from decision_ml.model_training import DecisionModelTrainer
from decision_ml.pipeline import DecisionMLPipeline

class TestDecisionDataPreprocessor(unittest.TestCase):
    """Test cases for DecisionDataPreprocessor."""
    
    def setUp(self):
        self.preprocessor = DecisionDataPreprocessor()
        
        # Sample job data
        self.sample_job_data = {
            "5185": {
                "informacoes_basicas": {
                    "titulo_vaga": "Python Developer",
                    "data_requicisao": "04-05-2021"
                },
                "perfil_vaga": {
                    "principais_atividades": "Develop Python applications with Django framework",
                    "competencia_tecnicas_e_comportamentais": "Python, Django, SQL, Git",
                    "nivel profissional": "Junior",
                    "nivel_academico": "Ensino Superior Completo",
                    "nivel_ingles": "Intermediário",
                    "nivel_espanhol": "Básico",
                    "pais": "Brasil",
                    "estado": "São Paulo",
                    "cidade": "São Paulo"
                }
            }
        }
        
        # Sample candidate data
        self.sample_candidate_data = [
            {
                "id": "John Doe",
                "perfil": "Python developer with 2 years experience in Django and SQL",
                "link": "https://linkedin.com/in/johndoe"
            }
        ]
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        text = "This is a TEST with SPECIAL characters!!! And   extra   spaces."
        cleaned = self.preprocessor.clean_text(text)
        
        self.assertIsInstance(cleaned, str)
        self.assertNotIn("!!!", cleaned)
        self.assertNotIn("   ", cleaned)
    
    def test_extract_skills_from_text(self):
        """Test skill extraction from text."""
        text = "I have experience with Python, Java, SQL, and React development"
        skills = self.preprocessor.extract_skills_from_text(text)
        
        self.assertIsInstance(skills, list)
        self.assertIn("python", skills)
        self.assertIn("java", skills)
        self.assertIn("sql", skills)
    
    def test_extract_experience_level(self):
        """Test experience level extraction."""
        junior_text = "I am a junior developer"
        senior_text = "I am a senior developer with 10 years experience"
        
        self.assertEqual(self.preprocessor.extract_experience_level(junior_text), "junior")
        self.assertEqual(self.preprocessor.extract_experience_level(senior_text), "senior")
    
    def test_preprocess_jobs_data(self):
        """Test job data preprocessing."""
        jobs_df = self.preprocessor.preprocess_jobs_data(self.sample_job_data)
        
        self.assertIsInstance(jobs_df, pd.DataFrame)
        self.assertEqual(len(jobs_df), 1)
        self.assertIn("job_id", jobs_df.columns)
        self.assertIn("description", jobs_df.columns)
        self.assertIn("skills", jobs_df.columns)
    
    def test_preprocess_candidates_data(self):
        """Test candidate data preprocessing."""
        candidates_df = self.preprocessor.preprocess_candidates_data(self.sample_candidate_data)
        
        self.assertIsInstance(candidates_df, pd.DataFrame)
        self.assertEqual(len(candidates_df), 1)
        self.assertIn("candidate_id", candidates_df.columns)
        self.assertIn("profile", candidates_df.columns)
        self.assertIn("skills", candidates_df.columns)

class TestDecisionFeatureEngineer(unittest.TestCase):
    """Test cases for DecisionFeatureEngineer."""
    
    def setUp(self):
        self.feature_engineer = DecisionFeatureEngineer()
        
        # Sample dataframes
        self.jobs_df = pd.DataFrame([{
            'job_id': '1',
            'title': 'Python Developer',
            'description': 'python django sql development',
            'skills': ['python', 'django', 'sql'],
            'experience_level': 'junior',
            'education_level': 'superior',
            'english_level': 'intermediario',
            'spanish_level': 'basico',
            'remote_work': True,
            'full_text': 'Python Django SQL development'
        }])
        
        self.candidates_df = pd.DataFrame([{
            'candidate_id': 'John Doe',
            'profile': 'python developer django experience',
            'skills': ['python', 'django'],
            'experience_level': 'junior',
            'education_level': 'superior',
            'english_level': 'intermediario',
            'spanish_level': 'basico',
            'full_text': 'Python developer with Django experience'
        }])
    
    def test_calculate_skill_match_score(self):
        """Test skill matching score calculation."""
        job_skills = ['python', 'django', 'sql']
        candidate_skills = ['python', 'django']
        
        score = self.feature_engineer.calculate_skill_match_score(job_skills, candidate_skills)
        
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0)
        self.assertLessEqual(score, 100)
    
    def test_calculate_experience_compatibility(self):
        """Test experience compatibility calculation."""
        score = self.feature_engineer.calculate_experience_compatibility('junior', 'junior')
        self.assertEqual(score, 100.0)
        
        score = self.feature_engineer.calculate_experience_compatibility('junior', 'pleno')
        self.assertEqual(score, 80.0)
    
    def test_create_matching_features(self):
        """Test matching features creation."""
        matching_df = self.feature_engineer.create_matching_features(self.jobs_df, self.candidates_df)
        
        self.assertIsInstance(matching_df, pd.DataFrame)
        self.assertEqual(len(matching_df), 1)  # 1 job x 1 candidate
        self.assertIn('skill_match_score', matching_df.columns)
        self.assertIn('experience_compatibility', matching_df.columns)
        self.assertIn('overall_score', matching_df.columns)
    
    def test_create_binary_labels(self):
        """Test binary label creation."""
        matching_df = pd.DataFrame([{
            'overall_score': 80.0
        }, {
            'overall_score': 60.0
        }])
        
        labeled_df = self.feature_engineer.create_binary_labels(matching_df, threshold=70.0)
        
        self.assertIn('is_good_match', labeled_df.columns)
        self.assertEqual(labeled_df.iloc[0]['is_good_match'], 1)
        self.assertEqual(labeled_df.iloc[1]['is_good_match'], 0)

class TestDecisionModelTrainer(unittest.TestCase):
    """Test cases for DecisionModelTrainer."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_trainer = DecisionModelTrainer(self.temp_dir)
        
        # Create sample training data
        np.random.seed(42)
        self.X = np.random.rand(100, 10)
        self.y = np.random.randint(0, 2, 100)
    
    def tearDown(self):
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_prepare_training_data(self):
        """Test training data preparation."""
        matching_df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'is_good_match': [1, 0, 1]
        })
        
        feature_columns = ['feature1', 'feature2']
        X, y = self.model_trainer.prepare_training_data(matching_df, feature_columns)
        
        self.assertEqual(X.shape, (3, 2))
        self.assertEqual(len(y), 3)
    
    @patch('decision_ml.model_training.GridSearchCV')
    def test_train_single_model(self, mock_grid_search):
        """Test single model training."""
        # Mock GridSearchCV
        mock_estimator = MagicMock()
        mock_estimator.predict.return_value = np.array([1, 0, 1])
        mock_estimator.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
        
        mock_grid_search.return_value.best_estimator_ = mock_estimator
        mock_grid_search.return_value.best_params_ = {'n_estimators': 100}
        mock_grid_search.return_value.fit.return_value = None
        
        X_train, X_val = self.X[:80], self.X[80:]
        y_train, y_val = self.y[:80], self.y[80:]
        
        results = self.model_trainer.train_single_model(
            'random_forest', X_train, y_train, X_val, y_val
        )
        
        self.assertIn('model', results)
        self.assertIn('accuracy', results)
        self.assertIn('roc_auc', results)
    
    def test_get_model_summary(self):
        """Test model summary generation."""
        summary = self.model_trainer.get_model_summary()
        self.assertIn('message', summary)

class TestDecisionMLPipeline(unittest.TestCase):
    """Test cases for DecisionMLPipeline."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = DecisionMLPipeline(self.temp_dir)
        
        # Create temporary data files
        self.jobs_file = os.path.join(self.temp_dir, 'jobs.json')
        self.candidates_file = os.path.join(self.temp_dir, 'candidates.json')
        
        jobs_data = {
            "1": {
                "informacoes_basicas": {"titulo_vaga": "Python Developer"},
                "perfil_vaga": {
                    "principais_atividades": "Python development",
                    "competencia_tecnicas_e_comportamentais": "Python, Django",
                    "nivel profissional": "Junior",
                    "nivel_academico": "Superior",
                    "nivel_ingles": "Intermediário",
                    "nivel_espanhol": "Básico",
                    "pais": "Brasil",
                    "estado": "SP",
                    "cidade": "São Paulo"
                }
            }
        }
        
        candidates_data = [
            {
                "id": "John Doe",
                "perfil": "Python developer with Django experience",
                "link": "https://linkedin.com/in/johndoe"
            }
        ]
        
        with open(self.jobs_file, 'w', encoding='utf-8') as f:
            json.dump(jobs_data, f)
        
        with open(self.candidates_file, 'w', encoding='utf-8') as f:
            json.dump(candidates_data, f)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_data(self):
        """Test data loading."""
        self.pipeline.load_data(self.jobs_file, self.candidates_file)
        
        self.assertIsNotNone(self.pipeline.jobs_df)
        self.assertIsNotNone(self.pipeline.candidates_df)
        self.assertEqual(len(self.pipeline.jobs_df), 1)
        self.assertEqual(len(self.pipeline.candidates_df), 1)
    
    def test_create_features(self):
        """Test feature creation."""
        self.pipeline.load_data(self.jobs_file, self.candidates_file)
        self.pipeline.create_features()
        
        self.assertIsNotNone(self.pipeline.matching_df)
        self.assertIsNotNone(self.pipeline.feature_columns)
        self.assertGreater(len(self.pipeline.feature_columns), 0)
    
    def test_get_pipeline_summary(self):
        """Test pipeline summary generation."""
        summary = self.pipeline.get_pipeline_summary()
        
        self.assertIn('pipeline_status', summary)
        self.assertIn('data_summary', summary)

if __name__ == '__main__':
    unittest.main()