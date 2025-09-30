"""
Vercel-optimized version of the Decision ML app.
Uses lightweight fallback implementation to avoid serverless function size limits.
"""

from fastapi import Form, BackgroundTasks
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil, os, json, time
from datetime import datetime
from Matching.preparingJobs import load_and_filter_jobs, transform_jobs
from Matching.pipeline import match_jobs_candidates

# Import lightweight fallback
from decision_ml_fallback import fallback_pipeline

app = FastAPI(
    title="hybridResources + Decision ML API",
    description="Job-candidate matching API with advanced ML capabilities",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables - using fallback mode for Vercel
DECISION_ML_AVAILABLE = True
DECISION_ML_MODE = "fallback"
decision_pipeline = None
decision_training_status = {"status": "not_started", "message": "No training initiated"}

@app.get("/", response_class=HTMLResponse)
def root():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>hybridResources + Decision ML API</h1><p>Advanced job-candidate matching system</p>", status_code=200)

@app.get("/health")
def health_check():
    health_status = {
        "status": "healthy", 
        "message": "API de matching ativa",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "components": {
            "original_matching": "active",
            "decision_ml": "available",
            "decision_ml_mode": "fallback_vercel_optimized",
            "decision_pipeline_trained": True
        }
    }
    
    try:
        health_status["decision_ml_health"] = fallback_pipeline.get_health_status()
    except:
        pass
    
    return health_status

@app.post("/match_vaga")
async def match_vaga_text(descricao: str = Form(...)):
    try:
        # 1. Monta objeto de vaga temporário
        vaga = {"id": "vaga_unica", "descricao": descricao}

        # 2. Carrega candidatos
        candidates_path = "JSONs/candidates.json"
        if not os.path.exists(candidates_path):
            return JSONResponse({"erro": "Arquivo de candidatos não encontrado."}, status_code=400)
        
        with open(candidates_path, "r", encoding="utf-8") as f:
            candidates = json.load(f)

        # 3. Aplica o matching
        res = match_jobs_candidates([vaga], candidates)

        # 4. Monta resposta: top 3 candidatos para a vaga
        match = res["top_matches"][0]
        top_candidatos = [
            {"candidato": c["cand_id"], "score": c["match_score"]}
            for c in match["top"]
        ]
        return {"vaga": descricao, "top_candidatos": top_candidatos}
    
    except Exception as e:
        return JSONResponse({"erro": "Erro interno do servidor", "detalhes": str(e)}, status_code=500)

@app.post("/match_vagas")
async def match_vagas(file: UploadFile = File(...)):
    vagas_path = "/tmp/vagas.json"
    try:
        # 1. Recebe o JSON e salva como vagas.json
        with open(vagas_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Faz as transformações corretas no JSON
        filtered_jobs = load_and_filter_jobs()
        if not filtered_jobs:
            return JSONResponse({"erro": "Erro ao carregar ou filtrar vagas."}, status_code=400)
        jobs_list = transform_jobs(filtered_jobs)

        # 3. Carrega candidatos
        candidates_path = "JSONs/candidates.json"
        if not os.path.exists(candidates_path):
            return JSONResponse({"erro": "Arquivo de candidatos não encontrado."}, status_code=400)
        
        with open(candidates_path, "r", encoding="utf-8") as f:
            candidates = json.load(f)

        # 4. Aplica o matching
        res = match_jobs_candidates(jobs_list, candidates)

        # 5. Monta resposta: top 3 candidatos para cada vaga
        top_matches = []
        for match in res["top_matches"]:
            top_matches.append({
                "vaga": match["job_id"],
                "top_candidatos": [
                    {"candidato": c["cand_id"], "score": c["match_score"]}
                    for c in match["top"]
                ]
            })

        return {"top_matches": top_matches}
    
    except Exception as e:
        return JSONResponse({"erro": "Erro interno do servidor", "detalhes": str(e)}, status_code=500)
    
    finally:
        # 6. Apaga o arquivo temporário de vagas
        if os.path.exists(vagas_path):
            try:
                os.remove(vagas_path)
            except:
                pass

# Decision ML Endpoints (Fallback Mode)
@app.post("/decision/train")
async def train_decision_model(background_tasks: BackgroundTasks, 
                             jobs_path: str = "JSONs_DECISION/vagas_padrao.json",
                             candidates_path: str = "JSONs_DECISION/candidates.json"):
    """Train Decision ML model (demonstration mode)."""
    global decision_training_status
    
    # Simulate training process
    decision_training_status = {"status": "training", "message": "Decision ML model training in progress (demo mode)..."}
    
    def simulate_training():
        global decision_training_status
        try:
            # Simulate training time
            time.sleep(2)
            
            decision_training_status = {
                "status": "completed", 
                "message": "Decision ML model training completed successfully (demo mode)",
                "timestamp": datetime.now().isoformat(),
                "training_time_seconds": 2.0,
                "best_model": "fallback_demonstration",
                "best_score": 0.85
            }
            
        except Exception as e:
            decision_training_status = {
                "status": "failed", 
                "message": f"Training simulation failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    # Start training simulation in background
    background_tasks.add_task(simulate_training)
    
    return {
        "message": "Decision ML model training started (demonstration mode)",
        "status": "training",
        "jobs_path": jobs_path,
        "candidates_path": candidates_path,
        "mode": "demonstration"
    }

@app.get("/decision/status")
async def get_decision_training_status():
    """Get Decision ML training status."""
    return fallback_pipeline.get_training_status()

@app.post("/decision/predict")
async def predict_decision_matches(job_description: str = Form(...), top_k: int = Form(3)):
    """Make advanced predictions using Decision ML model (demonstration mode)."""
    try:
        start_time = time.time()
        
        # Use fallback implementation
        matches = fallback_pipeline.predict_matches(job_description, top_k)
        
        top_candidatos = []
        for match in matches:
            top_candidatos.append({
                "candidato": match['candidate_id'],
                "score": match['match_probability'],
                "detailed_scores": {
                    "overall_score": match['overall_score'],
                    "skill_match": match['skill_match_score'],
                    "experience_compatibility": match['experience_compatibility'],
                    "education_compatibility": match['education_compatibility'],
                    "language_compatibility": match['language_compatibility'],
                    "text_similarity": match['text_similarity']
                }
            })
        
        response_time = time.time() - start_time
        
        return {
            "vaga": job_description,
            "top_candidatos": top_candidatos,
            "model_used": fallback_pipeline.model_name,
            "response_time_ms": response_time * 1000,
            "timestamp": datetime.now().isoformat(),
            "mode": "demonstration"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decision ML prediction failed: {str(e)}")

@app.get("/decision/monitoring/health")
async def get_decision_monitoring_health():
    """Get Decision ML monitoring health status."""
    try:
        return fallback_pipeline.get_health_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring health: {str(e)}")

@app.get("/decision/monitoring/report")
async def get_decision_monitoring_report():
    """Get comprehensive Decision ML monitoring report."""
    try:
        return fallback_pipeline.generate_report()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate monitoring report: {str(e)}")

@app.get("/decision/models/summary")
async def get_decision_models_summary():
    """Get summary of trained Decision ML models."""
    return {
        "pipeline_status": "demonstration_mode",
        "mode": "vercel_optimized_fallback",
        "message": "Running in Vercel-optimized demonstration mode with basic matching algorithm",
        "data_summary": {
            "jobs_count": 50,
            "candidates_count": 5,
            "feature_count": 10
        },
        "deployment": {
            "platform": "vercel",
            "optimization": "lightweight_fallback",
            "function_size": "under_250mb"
        }
    }

# For Vercel compatibility
app_handler = app