"""
Feature engineering module for Decision recruitment ML pipeline.
Creates advanced features for job-candidate matching.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)

class DecisionFeatureEngineer:
    """
    Feature engineering class for creating advanced matching features.
    """
    
    def __init__(self):
        self.skill_weights = {
            'python': 1.5, 'java': 1.5, 'javascript': 1.4, 'sql': 1.3,
            'react': 1.3, 'angular': 1.3, 'aws': 1.4, 'docker': 1.2,
            'machine learning': 1.6, 'data science': 1.6, 'devops': 1.3,
            'agile': 1.1, 'scrum': 1.1, 'git': 1.1, 'linux': 1.2
        }
        
        self.experience_mapping = {
            'junior': 1, 'pleno': 2, 'senior': 3, 'especialista': 4, 'lead': 5
        }
        
        self.education_mapping = {
            'medio': 1, 'tecnico': 2, 'superior': 3, 'pos_graduacao': 4
        }
        
        self.language_mapping = {
            'basico': 1, 'intermediario': 2, 'avancado': 3, 'fluente': 4
        }
    
    def calculate_skill_match_score(self, job_skills: List[str], candidate_skills: List[str]) -> float:
        """Calculate weighted skill match score between job and candidate."""
        if not job_skills:
            return 0.0
        
        job_skills_set = set([skill.lower() for skill in job_skills])
        candidate_skills_set = set([skill.lower() for skill in candidate_skills])
        
        # Calculate weighted intersection
        matched_skills = job_skills_set.intersection(candidate_skills_set)
        
        if not matched_skills:
            return 0.0
        
        # Apply weights to matched skills
        weighted_score = 0.0
        total_weight = 0.0
        
        for skill in job_skills_set:
            weight = self.skill_weights.get(skill, 1.0)
            total_weight += weight
            
            if skill in matched_skills:
                weighted_score += weight
        
        return (weighted_score / total_weight) * 100 if total_weight > 0 else 0.0
    
    def calculate_experience_compatibility(self, job_level: str, candidate_level: str) -> float:
        """Calculate experience level compatibility score."""
        job_score = self.experience_mapping.get(job_level.lower(), 1)
        candidate_score = self.experience_mapping.get(candidate_level.lower(), 1)
        
        # Perfect match gets 100%, one level difference gets 80%, etc.
        diff = abs(job_score - candidate_score)
        
        if diff == 0:
            return 100.0
        elif diff == 1:
            return 80.0
        elif diff == 2:
            return 60.0
        else:
            return 40.0
    
    def calculate_education_compatibility(self, job_education: str, candidate_education: str) -> float:
        """Calculate education level compatibility score."""
        job_score = self.education_mapping.get(job_education.lower(), 3)
        candidate_score = self.education_mapping.get(candidate_education.lower(), 3)
        
        # Candidate education should be >= job requirement
        if candidate_score >= job_score:
            return 100.0
        elif candidate_score == job_score - 1:
            return 70.0
        else:
            return 40.0
    
    def calculate_language_compatibility(self, job_english: str, job_spanish: str, 
                                       candidate_english: str, candidate_spanish: str) -> float:
        """Calculate language compatibility score."""
        job_eng_score = self.language_mapping.get(job_english.lower(), 1)
        job_spa_score = self.language_mapping.get(job_spanish.lower(), 1)
        candidate_eng_score = self.language_mapping.get(candidate_english.lower(), 1)
        candidate_spa_score = self.language_mapping.get(candidate_spanish.lower(), 1)
        
        # Calculate compatibility for each language
        eng_compatibility = min(candidate_eng_score / job_eng_score, 1.0) * 100 if job_eng_score > 0 else 100.0
        spa_compatibility = min(candidate_spa_score / job_spa_score, 1.0) * 100 if job_spa_score > 0 else 100.0
        
        # Average the two language scores
        return (eng_compatibility + spa_compatibility) / 2
    
    def calculate_text_similarity(self, job_text: str, candidate_text: str) -> float:
        """Calculate semantic similarity between job description and candidate profile."""
        if not job_text or not candidate_text:
            return 0.0
        
        try:
            vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([job_text, candidate_text])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity * 100
        except:
            return 0.0
    
    def create_matching_features(self, jobs_df: pd.DataFrame, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive matching features for all job-candidate pairs."""
        logger.info("Creating matching features...")
        
        matching_data = []
        
        for _, job in jobs_df.iterrows():
            for _, candidate in candidates_df.iterrows():
                # Calculate various compatibility scores
                skill_score = self.calculate_skill_match_score(job['skills'], candidate['skills'])
                experience_score = self.calculate_experience_compatibility(job['experience_level'], candidate['experience_level'])
                education_score = self.calculate_education_compatibility(job['education_level'], candidate['education_level'])
                language_score = self.calculate_language_compatibility(
                    job['english_level'], job['spanish_level'],
                    candidate['english_level'], candidate['spanish_level']
                )
                text_similarity = self.calculate_text_similarity(job['full_text'], candidate['full_text'])
                
                # Additional features
                skill_count_job = len(job['skills'])
                skill_count_candidate = len(candidate['skills'])
                skill_ratio = skill_count_candidate / max(skill_count_job, 1)
                
                # Remote work compatibility
                remote_compatibility = 100.0 if job['remote_work'] else 80.0
                
                # Create feature vector
                features = {
                    'job_id': job['job_id'],
                    'candidate_id': candidate['candidate_id'],
                    'skill_match_score': skill_score,
                    'experience_compatibility': experience_score,
                    'education_compatibility': education_score,
                    'language_compatibility': language_score,
                    'text_similarity': text_similarity,
                    'skill_count_ratio': skill_ratio,
                    'remote_compatibility': remote_compatibility,
                    'job_skill_count': skill_count_job,
                    'candidate_skill_count': skill_count_candidate,
                    # Overall compatibility score (weighted average)
                    'overall_score': (
                        skill_score * 0.35 +
                        experience_score * 0.25 +
                        education_score * 0.15 +
                        language_score * 0.10 +
                        text_similarity * 0.15
                    )
                }
                
                matching_data.append(features)
        
        return pd.DataFrame(matching_data)
    
    def create_candidate_ranking_features(self, matching_df: pd.DataFrame) -> pd.DataFrame:
        """Create features for ranking candidates within each job."""
        logger.info("Creating candidate ranking features...")
        
        # Group by job and calculate relative rankings
        ranking_features = []
        
        for job_id in matching_df['job_id'].unique():
            job_matches = matching_df[matching_df['job_id'] == job_id].copy()
            
            # Calculate percentile ranks for each score
            job_matches['skill_rank'] = job_matches['skill_match_score'].rank(pct=True)
            job_matches['experience_rank'] = job_matches['experience_compatibility'].rank(pct=True)
            job_matches['education_rank'] = job_matches['education_compatibility'].rank(pct=True)
            job_matches['language_rank'] = job_matches['language_compatibility'].rank(pct=True)
            job_matches['text_similarity_rank'] = job_matches['text_similarity'].rank(pct=True)
            job_matches['overall_rank'] = job_matches['overall_score'].rank(pct=True)
            
            # Create composite ranking score
            job_matches['composite_rank'] = (
                job_matches['skill_rank'] * 0.35 +
                job_matches['experience_rank'] * 0.25 +
                job_matches['education_rank'] * 0.15 +
                job_matches['language_rank'] * 0.10 +
                job_matches['text_similarity_rank'] * 0.15
            )
            
            ranking_features.append(job_matches)
        
        return pd.concat(ranking_features, ignore_index=True)
    
    def create_binary_labels(self, matching_df: pd.DataFrame, threshold: float = 70.0) -> pd.DataFrame:
        """Create binary labels for classification (good match vs poor match)."""
        logger.info(f"Creating binary labels with threshold: {threshold}")
        
        matching_df = matching_df.copy()
        matching_df['is_good_match'] = (matching_df['overall_score'] >= threshold).astype(int)
        
        return matching_df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns for ML model."""
        return [
            'skill_match_score',
            'experience_compatibility', 
            'education_compatibility',
            'language_compatibility',
            'text_similarity',
            'skill_count_ratio',
            'remote_compatibility',
            'job_skill_count',
            'candidate_skill_count',
            'skill_rank',
            'experience_rank',
            'education_rank',
            'language_rank',
            'text_similarity_rank',
            'composite_rank'
        ]