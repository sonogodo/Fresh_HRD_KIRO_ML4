"""
Data preprocessing module for Decision recruitment ML pipeline.
Handles data cleaning, feature extraction, and preparation for model training.
"""

import pandas as pd
import numpy as np
import re
import json
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecisionDataPreprocessor:
    """
    Preprocessor for Decision recruitment data.
    Handles job descriptions, candidate profiles, and feature engineering.
    """
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.portuguese_stopwords = set([
            'de', 'da', 'do', 'em', 'com', 'e', 'a', 'o', 'para', 'por', 'um', 'uma', 
            'no', 'na', 'os', 'as', 'que', 'se', 'ou', 'mais', 'ser', 'ter', 'como',
            'trabalho', 'experiencia', 'conhecimento', 'habilidades', 'anos'
        ])
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text data."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove Portuguese stopwords
        words = text.split()
        words = [word for word in words if word not in self.portuguese_stopwords and len(word) > 2]
        
        return ' '.join(words)
    
    def extract_skills_from_text(self, text: str) -> List[str]:
        """Extract technical skills from text using pattern matching."""
        if not isinstance(text, str):
            return []
        
        # Common technical skills patterns
        skill_patterns = [
            r'\b(python|java|javascript|react|angular|vue|node|sql|mysql|postgresql|mongodb|aws|azure|gcp|docker|kubernetes|git|linux|windows|html|css|php|ruby|go|rust|scala|kotlin|swift|c\+\+|c#|\.net|spring|django|flask|tensorflow|pytorch|scikit-learn|pandas|numpy|matplotlib|tableau|power bi|excel|autocad|sap|oracle|salesforce|jira|confluence|agile|scrum|devops|ci/cd|jenkins|terraform|ansible)\b',
            r'\b(engenharia|civil|mecanica|eletrica|computacao|sistemas|dados|software|frontend|backend|fullstack|mobile|web|cloud|machine learning|data science|business intelligence|cybersecurity|redes|infraestrutura|banco de dados|analise|desenvolvimento|programacao|arquitetura|design|ux|ui)\b'
        ]
        
        skills = []
        text_lower = text.lower()
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text_lower)
            skills.extend(matches)
        
        return list(set(skills))
    
    def extract_experience_level(self, text: str) -> str:
        """Extract experience level from text."""
        if not isinstance(text, str):
            return "junior"
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['senior', 'sênior', 'especialista', 'lead', 'principal']):
            return "senior"
        elif any(word in text_lower for word in ['pleno', 'intermediario', 'mid-level']):
            return "pleno"
        else:
            return "junior"
    
    def extract_education_level(self, text: str) -> str:
        """Extract education level from text."""
        if not isinstance(text, str):
            return "superior"
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['mestrado', 'master', 'doutorado', 'phd', 'pos-graduacao']):
            return "pos_graduacao"
        elif any(word in text_lower for word in ['superior', 'graduacao', 'bacharelado', 'licenciatura']):
            return "superior"
        elif any(word in text_lower for word in ['tecnico', 'tecnólogo']):
            return "tecnico"
        else:
            return "medio"
    
    def extract_language_skills(self, text: str) -> Dict[str, str]:
        """Extract language proficiency from text."""
        if not isinstance(text, str):
            return {"ingles": "basico", "espanhol": "basico"}
        
        text_lower = text.lower()
        languages = {"ingles": "basico", "espanhol": "basico"}
        
        # English proficiency
        if any(word in text_lower for word in ['fluente', 'fluent', 'avancado', 'advanced']):
            if 'ingles' in text_lower or 'english' in text_lower:
                languages["ingles"] = "avancado"
        elif any(word in text_lower for word in ['intermediario', 'intermediate']):
            if 'ingles' in text_lower or 'english' in text_lower:
                languages["ingles"] = "intermediario"
        
        # Spanish proficiency
        if any(word in text_lower for word in ['fluente', 'fluent', 'avancado', 'advanced']):
            if 'espanhol' in text_lower or 'spanish' in text_lower:
                languages["espanhol"] = "avancado"
        elif any(word in text_lower for word in ['intermediario', 'intermediate']):
            if 'espanhol' in text_lower or 'spanish' in text_lower:
                languages["espanhol"] = "intermediario"
        
        return languages
    
    def preprocess_jobs_data(self, jobs_data: Dict) -> pd.DataFrame:
        """Preprocess jobs data from Decision format."""
        logger.info("Preprocessing jobs data...")
        
        processed_jobs = []
        
        for job_id, job_info in jobs_data.items():
            try:
                # Extract basic information
                basic_info = job_info.get('informacoes_basicas', {})
                profile_info = job_info.get('perfil_vaga', {})
                
                # Combine text fields for analysis
                job_text = " ".join([
                    basic_info.get('titulo_vaga', ''),
                    profile_info.get('principais_atividades', ''),
                    profile_info.get('competencia_tecnicas_e_comportamentais', ''),
                    profile_info.get('demais_observacoes', '')
                ])
                
                # Clean text
                cleaned_text = self.clean_text(job_text)
                
                # Extract features
                skills = self.extract_skills_from_text(job_text)
                experience_level = profile_info.get('nivel profissional', 'junior').lower()
                education_level = self.extract_education_level(profile_info.get('nivel_academico', ''))
                
                processed_job = {
                    'job_id': job_id,
                    'title': basic_info.get('titulo_vaga', ''),
                    'description': cleaned_text,
                    'skills': skills,
                    'experience_level': experience_level,
                    'education_level': education_level,
                    'location': f"{profile_info.get('cidade', '')} - {profile_info.get('estado', '')}",
                    'english_level': profile_info.get('nivel_ingles', 'basico').lower(),
                    'spanish_level': profile_info.get('nivel_espanhol', 'basico').lower(),
                    'remote_work': 'remoto' in job_text.lower(),
                    'contract_type': basic_info.get('tipo_contratacao', ''),
                    'client': basic_info.get('cliente', ''),
                    'full_text': job_text
                }
                
                processed_jobs.append(processed_job)
                
            except Exception as e:
                logger.error(f"Error processing job {job_id}: {str(e)}")
                continue
        
        return pd.DataFrame(processed_jobs)
    
    def preprocess_candidates_data(self, candidates_data: List[Dict]) -> pd.DataFrame:
        """Preprocess candidates data."""
        logger.info("Preprocessing candidates data...")
        
        processed_candidates = []
        
        for candidate in candidates_data:
            try:
                candidate_id = candidate.get('id', '')
                profile_text = candidate.get('perfil', '')
                
                if not profile_text:
                    continue
                
                # Clean text
                cleaned_text = self.clean_text(profile_text)
                
                # Extract features
                skills = self.extract_skills_from_text(profile_text)
                experience_level = self.extract_experience_level(profile_text)
                education_level = self.extract_education_level(profile_text)
                language_skills = self.extract_language_skills(profile_text)
                
                processed_candidate = {
                    'candidate_id': candidate_id,
                    'profile': cleaned_text,
                    'skills': skills,
                    'experience_level': experience_level,
                    'education_level': education_level,
                    'english_level': language_skills['ingles'],
                    'spanish_level': language_skills['espanhol'],
                    'linkedin_url': candidate.get('link', ''),
                    'full_text': profile_text
                }
                
                processed_candidates.append(processed_candidate)
                
            except Exception as e:
                logger.error(f"Error processing candidate {candidate.get('id', 'unknown')}: {str(e)}")
                continue
        
        return pd.DataFrame(processed_candidates)
    
    def create_features_matrix(self, jobs_df: pd.DataFrame, candidates_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Create feature matrices for ML model training."""
        logger.info("Creating feature matrices...")
        
        # Combine all text for TF-IDF fitting
        all_texts = list(jobs_df['description']) + list(candidates_df['profile'])
        
        # Fit TF-IDF vectorizer
        self.tfidf_vectorizer.fit(all_texts)
        
        # Transform job descriptions
        job_tfidf = self.tfidf_vectorizer.transform(jobs_df['description'])
        
        # Transform candidate profiles
        candidate_tfidf = self.tfidf_vectorizer.transform(candidates_df['profile'])
        
        # Get feature names
        feature_names = self.tfidf_vectorizer.get_feature_names_out().tolist()
        
        return job_tfidf.toarray(), candidate_tfidf.toarray(), feature_names