"""
Lightweight fallback implementation for Decision ML optimized for Vercel deployment.
Provides basic functionality for demonstration purposes without heavy dependencies.
"""

import json
import random
import re
from datetime import datetime
from typing import List, Dict, Any

class DecisionMLFallback:
    """
    Fallback implementation that provides basic matching functionality
    when the full Decision ML pipeline is not available.
    """
    
    def __init__(self):
        self.is_trained = False
        self.model_name = "fallback_basic_matcher"
        
        # Sample candidate data for demonstration
        self.sample_candidates = [
            {
                "candidate_id": "João Silva",
                "skills": ["python", "django", "sql", "git"],
                "experience": "junior",
                "score_base": 0.85
            },
            {
                "candidate_id": "Maria Santos", 
                "skills": ["javascript", "react", "node", "mongodb"],
                "experience": "pleno",
                "score_base": 0.78
            },
            {
                "candidate_id": "Pedro Costa",
                "skills": ["java", "spring", "mysql", "docker"],
                "experience": "junior", 
                "score_base": 0.72
            },
            {
                "candidate_id": "Ana Oliveira",
                "skills": ["python", "machine learning", "pandas", "scikit-learn"],
                "experience": "senior",
                "score_base": 0.91
            },
            {
                "candidate_id": "Carlos Ferreira",
                "skills": ["php", "laravel", "mysql", "vue"],
                "experience": "pleno",
                "score_base": 0.68
            }
        ]
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract basic keywords from job description."""
        keywords = []
        text_lower = text.lower()
        
        # Common tech keywords
        tech_keywords = [
            'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node',
            'django', 'flask', 'spring', 'laravel', 'php', 'sql', 'mysql',
            'postgresql', 'mongodb', 'docker', 'kubernetes', 'aws', 'azure',
            'git', 'machine learning', 'data science', 'pandas', 'scikit-learn'
        ]
        
        for keyword in tech_keywords:
            if keyword in text_lower:
                keywords.append(keyword)
        
        return keywords
    
    def calculate_match_score(self, job_keywords: List[str], candidate: Dict) -> float:
        """Calculate basic match score between job and candidate."""
        if not job_keywords:
            return random.uniform(0.3, 0.7)
        
        # Calculate keyword overlap
        candidate_skills = [skill.lower() for skill in candidate.get('skills', [])]
        job_keywords_lower = [kw.lower() for kw in job_keywords]
        
        overlap = len(set(candidate_skills) & set(job_keywords_lower))
        max_possible = max(len(job_keywords_lower), len(candidate_skills))
        
        if max_possible == 0:
            base_score = 0.5
        else:
            base_score = overlap / max_possible
        
        # Add some randomness and candidate base score
        final_score = (base_score * 0.7) + (candidate['score_base'] * 0.3)
        
        # Add small random variation
        final_score += random.uniform(-0.05, 0.05)
        
        return max(0.1, min(0.95, final_score))
    
    def predict_matches(self, job_description: str, top_k: int = 3) -> List[Dict]:
        """Predict top matches for a job description."""
        job_keywords = self.extract_keywords(job_description)
        
        # Calculate scores for all candidates
        scored_candidates = []
        for candidate in self.sample_candidates:
            score = self.calculate_match_score(job_keywords, candidate)
            
            scored_candidates.append({
                "candidate_id": candidate["candidate_id"],
                "match_probability": score,
                "overall_score": score * 100,
                "skill_match_score": score * 100 + random.uniform(-10, 10),
                "experience_compatibility": random.uniform(70, 100),
                "education_compatibility": random.uniform(80, 100),
                "language_compatibility": random.uniform(60, 90),
                "text_similarity": score * 80 + random.uniform(-15, 15)
            })
        
        # Sort by score and return top K
        scored_candidates.sort(key=lambda x: x["match_probability"], reverse=True)
        return scored_candidates[:top_k]
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get mock training status."""
        return {
            "status": "completed",
            "message": "Fallback mode - basic matching active",
            "timestamp": datetime.now().isoformat(),
            "best_model": self.model_name,
            "best_score": 0.75,
            "pipeline_summary": {
                "jobs_count": 50,
                "candidates_count": len(self.sample_candidates),
                "job_candidate_pairs": 250
            }
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get mock health status."""
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "metrics": {
                "total_predictions_24h": random.randint(10, 50),
                "avg_response_time_ms": random.uniform(100, 300),
                "drift_alerts_7d": 0,
                "performance_baseline": 0.75
            },
            "alerts": {
                "active_drift_alerts": 0,
                "recent_errors": 0
            },
            "mode": "fallback"
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate mock monitoring report."""
        return {
            "report_timestamp": datetime.now().isoformat(),
            "system_health": self.get_health_status(),
            "summary": {
                "total_requests": random.randint(100, 500),
                "total_drift_alerts": 0,
                "monitoring_period_days": 7,
                "mode": "fallback_demonstration"
            },
            "message": "This is a demonstration mode. Full ML capabilities require proper training data and model deployment."
        }

# Global fallback instance
fallback_pipeline = DecisionMLFallback()