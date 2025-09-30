"""
FastAPI application for Decision recruitment ML system.
Provides endpoints for training models and making predictions.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import logging
import os
from datetime import datetime

from decision_ml.pipeline import DecisionMLPipeline, run_complete_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Decision Recruitment ML API",
    description="Machine Learning API for job-candidate matching in recruitment",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = None
training_status = {"status": "not_started", "message": "No training initiated"}

# Pydantic models for API
class JobData(BaseModel):
    informacoes_basicas: Dict[str, Any]
    perfil_vaga: Dict[str, Any]
    beneficios: Optional[Dict[str, Any]] = {}

class CandidateData(BaseModel):
    id: str
    perfil: str
    link: Optional[str] = ""

class PredictionRequest(BaseModel):
    job_data: Optional[JobData] = None
    candidate_data: Optional[List[CandidateData]] = None
    top_k: Optional[int] = 3

class TrainingRequest(BaseModel):
    jobs_path: Optional[str] = "JSONs_DECISION/vagas_padrao.json"
    candidates_path: Optional[str] = "JSONs_DECISION/candidates.json"
    threshold: Optional[float] = 70.0

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Decision Recruitment ML API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "/health": "Health check",
            "/train": "Train ML models",
            "/predict": "Make predictions",
            "/status": "Get training status",
            "/models/summary": "Get model summary"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global pipeline
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "pipeline_trained": pipeline is not None and pipeline.is_trained,
        "api_version": "1.0.0"
    }

@app.post("/train")
async def train_models(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train ML models with provided data."""
    global pipeline, training_status
    
    # Check if files exist
    if not os.path.exists(request.jobs_path):
        raise HTTPException(status_code=400, detail=f"Jobs file not found: {request.jobs_path}")
    
    if not os.path.exists(request.candidates_path):
        raise HTTPException(status_code=400, detail=f"Candidates file not found: {request.candidates_path}")
    
    # Update training status
    training_status = {"status": "training", "message": "Model training in progress..."}
    
    def train_pipeline():
        global pipeline, training_status
        try:
            logger.info("Starting model training...")
            pipeline = run_complete_pipeline(request.jobs_path, request.candidates_path)
            training_status = {
                "status": "completed", 
                "message": "Model training completed successfully",
                "timestamp": datetime.now().isoformat()
            }
            logger.info("Model training completed successfully")
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            training_status = {
                "status": "failed", 
                "message": f"Training failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    # Start training in background
    background_tasks.add_task(train_pipeline)
    
    return {
        "message": "Model training started",
        "status": "training",
        "jobs_path": request.jobs_path,
        "candidates_path": request.candidates_path
    }

@app.get("/status")
async def get_training_status():
    """Get current training status."""
    global training_status, pipeline
    
    status_response = training_status.copy()
    
    if pipeline and pipeline.is_trained:
        status_response["pipeline_summary"] = pipeline.get_pipeline_summary()
    
    return status_response

@app.post("/predict")
async def predict_matches(request: PredictionRequest):
    """Make job-candidate match predictions."""
    global pipeline
    
    if pipeline is None or not pipeline.is_trained:
        raise HTTPException(
            status_code=400, 
            detail="Model not trained. Please train the model first using /train endpoint."
        )
    
    try:
        # Convert Pydantic models to dictionaries if provided
        job_data = request.job_data.dict() if request.job_data else None
        candidate_data = [c.dict() for c in request.candidate_data] if request.candidate_data else None
        
        # Make predictions
        predictions = pipeline.predict_matches(
            job_data=job_data,
            candidate_data=candidate_data,
            top_k=request.top_k
        )
        
        return {
            "predictions": predictions,
            "timestamp": datetime.now().isoformat(),
            "model_used": pipeline.model_trainer.best_model_name
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/models/summary")
async def get_model_summary():
    """Get summary of trained models."""
    global pipeline
    
    if pipeline is None:
        return {"message": "No pipeline initialized"}
    
    return pipeline.get_pipeline_summary()

@app.post("/predict/single_job")
async def predict_single_job(job_description: str, top_k: int = 3):
    """Predict matches for a single job description."""
    global pipeline
    
    if pipeline is None or not pipeline.is_trained:
        raise HTTPException(
            status_code=400, 
            detail="Model not trained. Please train the model first using /train endpoint."
        )
    
    try:
        # Create a temporary job data structure
        job_data = {
            "informacoes_basicas": {
                "titulo_vaga": "Temporary Job",
                "data_requicisao": datetime.now().strftime("%d-%m-%Y")
            },
            "perfil_vaga": {
                "principais_atividades": job_description,
                "competencia_tecnicas_e_comportamentais": job_description,
                "nivel profissional": "junior",
                "nivel_academico": "Ensino Superior Completo",
                "nivel_ingles": "Intermediário",
                "nivel_espanhol": "Básico",
                "pais": "Brasil",
                "estado": "São Paulo",
                "cidade": "São Paulo"
            }
        }
        
        # Make predictions
        predictions = pipeline.predict_matches(
            job_data=job_data,
            candidate_data=None,
            top_k=top_k
        )
        
        return {
            "job_description": job_description,
            "predictions": predictions,
            "timestamp": datetime.now().isoformat(),
            "model_used": pipeline.model_trainer.best_model_name
        }
        
    except Exception as e:
        logger.error(f"Single job prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/candidates/count")
async def get_candidates_count():
    """Get count of available candidates."""
    global pipeline
    
    if pipeline is None:
        return {"candidates_count": 0, "message": "Pipeline not initialized"}
    
    count = len(pipeline.candidates_df) if pipeline.candidates_df is not None else 0
    return {"candidates_count": count}

@app.get("/jobs/count")
async def get_jobs_count():
    """Get count of available jobs."""
    global pipeline
    
    if pipeline is None:
        return {"jobs_count": 0, "message": "Pipeline not initialized"}
    
    count = len(pipeline.jobs_df) if pipeline.jobs_df is not None else 0
    return {"jobs_count": count}

@app.post("/evaluate")
async def evaluate_model():
    """Evaluate the trained model."""
    global pipeline
    
    if pipeline is None or not pipeline.is_trained:
        raise HTTPException(
            status_code=400, 
            detail="Model not trained. Please train the model first using /train endpoint."
        )
    
    try:
        evaluation_results = pipeline.evaluate_pipeline()
        
        # Convert numpy types to Python types for JSON serialization
        serializable_results = {
            "accuracy": float(evaluation_results["accuracy"]),
            "roc_auc": float(evaluation_results["roc_auc"]),
            "classification_report": evaluation_results["classification_report"],
            "confusion_matrix": evaluation_results["confusion_matrix"].tolist(),
            "timestamp": datetime.now().isoformat()
        }
        
        return serializable_results
        
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)