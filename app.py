from fastapi import Form, BackgroundTasks
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil, os, json, time
from datetime import datetime
from Matching.preparingJobs import load_and_filter_jobs, transform_jobs
from Matching.pipeline import match_jobs_candidates

# Initialize Decision ML components
try:
    # Run initialization
    from init_decision_ml import initialize
    initialize()
    
    # Import Decision ML components
    from decision_ml.pipeline import DecisionMLPipeline, run_complete_pipeline
    from decision_ml.monitoring import monitor
    DECISION_ML_AVAILABLE = True
    DECISION_ML_MODE = "full"
    print("✅ Decision ML components loaded successfully")
except ImportError as e:
    DECISION_ML_AVAILABLE = True  # Still available in fallback mode
    DECISION_ML_MODE = "fallback"
    print(f"⚠️ Using Decision ML fallback mode: {e}")
    from decision_ml_fallback import fallback_pipeline
except Exception as e:
    DECISION_ML_AVAILABLE = True  # Still available in fallback mode
    DECISION_ML_MODE = "fallback"
    print(f"⚠️ Using Decision ML fallback mode: {e}")
    from decision_ml_fallback import fallback_pipeline

app = FastAPI(
    title="hybridResources + Decision ML API",
    description="Job-candidate matching API with advanced ML capabilities",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables for Decision ML
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
    global decision_pipeline
    
    health_status = {
        "status": "healthy", 
        "message": "API de matching ativa",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "components": {
            "original_matching": "active",
            "decision_ml": "available" if DECISION_ML_AVAILABLE else "not_available",
            "decision_ml_mode": DECISION_ML_MODE if DECISION_ML_AVAILABLE else "unavailable",
            "decision_pipeline_trained": (
                decision_pipeline is not None and decision_pipeline.is_trained 
                if DECISION_ML_MODE == "full" else True
            )
        }
    }
    
    if DECISION_ML_AVAILABLE and decision_pipeline:
        try:
            health_status["decision_ml_health"] = monitor.get_system_health()
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
                pass  # Ignore cleanup errors

# Decision ML Endpoints
if DECISION_ML_AVAILABLE:
    
    @app.post("/decision/train")
    async def train_decision_model(background_tasks: BackgroundTasks, 
                                 jobs_path: str = "JSONs_DECISION/vagas_padrao.json",
                                 candidates_path: str = "JSONs_DECISION/candidates.json"):
        """Train Decision ML model with advanced features."""
        global decision_pipeline, decision_training_status
        
        # Check if files exist
        if not os.path.exists(jobs_path):
            raise HTTPException(status_code=400, detail=f"Jobs file not found: {jobs_path}")
        
        if not os.path.exists(candidates_path):
            raise HTTPException(status_code=400, detail=f"Candidates file not found: {candidates_path}")
        
        # Update training status
        decision_training_status = {"status": "training", "message": "Decision ML model training in progress..."}
        
        def train_pipeline():
            global decision_pipeline, decision_training_status
            try:
                start_time = time.time()
                decision_pipeline = run_complete_pipeline(jobs_path, candidates_path)
                training_time = time.time() - start_time
                
                # Log training event
                training_results = {
                    'best_model_name': decision_pipeline.model_trainer.best_model_name,
                    'best_score': decision_pipeline.model_trainer.models[decision_pipeline.model_trainer.best_model_name]['roc_auc'],
                    'training_time_seconds': training_time,
                    'all_results': decision_pipeline.model_trainer.models
                }
                
                monitor.log_training_event(training_results)
                
                decision_training_status = {
                    "status": "completed", 
                    "message": "Decision ML model training completed successfully",
                    "timestamp": datetime.now().isoformat(),
                    "training_time_seconds": training_time,
                    "best_model": training_results['best_model_name'],
                    "best_score": training_results['best_score']
                }
                
            except Exception as e:
                monitor.log_error('decision_training_error', str(e))
                decision_training_status = {
                    "status": "failed", 
                    "message": f"Decision ML training failed: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
        
        # Start training in background
        background_tasks.add_task(train_pipeline)
        
        return {
            "message": "Decision ML model training started",
            "status": "training",
            "jobs_path": jobs_path,
            "candidates_path": candidates_path
        }
    
    @app.get("/decision/status")
    async def get_decision_training_status():
        """Get Decision ML training status."""
        global decision_training_status, decision_pipeline
        
        if DECISION_ML_MODE == "fallback":
            return fallback_pipeline.get_training_status()
        
        status_response = decision_training_status.copy()
        
        if decision_pipeline and decision_pipeline.is_trained:
            status_response["pipeline_summary"] = decision_pipeline.get_pipeline_summary()
        
        return status_response
    
    @app.post("/decision/predict")
    async def predict_decision_matches(job_description: str = Form(...), top_k: int = Form(3)):
        """Make advanced predictions using Decision ML model."""
        global decision_pipeline
        
        try:
            start_time = time.time()
            
            if DECISION_ML_MODE == "fallback":
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
            
            else:
                # Use full ML pipeline
                if decision_pipeline is None or not decision_pipeline.is_trained:
                    raise HTTPException(
                        status_code=400, 
                        detail="Decision ML model not trained. Please train the model first using /decision/train endpoint."
                    )
                
                # Create temporary job data structure
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
                predictions = decision_pipeline.predict_matches(
                    job_data=job_data,
                    candidate_data=None,
                    top_k=top_k
                )
                
                response_time = time.time() - start_time
                
                # Log prediction request
                monitor.log_prediction_request(
                    {"job_description": job_description, "top_k": top_k},
                    predictions,
                    response_time
                )
                
                # Format response similar to original API
                if predictions:
                    top_candidatos = []
                    for match in predictions[0]['top_matches']:
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
                    
                    return {
                        "vaga": job_description,
                        "top_candidatos": top_candidatos,
                        "model_used": decision_pipeline.model_trainer.best_model_name,
                        "response_time_ms": response_time * 1000,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {"vaga": job_description, "top_candidatos": [], "message": "No matches found"}
            
        except HTTPException:
            raise
        except Exception as e:
            if DECISION_ML_MODE == "full":
                monitor.log_error('decision_prediction_error', str(e), {"job_description": job_description})
            raise HTTPException(status_code=500, detail=f"Decision ML prediction failed: {str(e)}")
    
    @app.get("/decision/monitoring/health")
    async def get_decision_monitoring_health():
        """Get Decision ML monitoring health status."""
        try:
            if DECISION_ML_MODE == "fallback":
                return fallback_pipeline.get_health_status()
            return monitor.get_system_health()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get monitoring health: {str(e)}")
    
    @app.get("/decision/monitoring/report")
    async def get_decision_monitoring_report():
        """Get comprehensive Decision ML monitoring report."""
        try:
            if DECISION_ML_MODE == "fallback":
                return fallback_pipeline.generate_report()
            return monitor.generate_monitoring_report()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate monitoring report: {str(e)}")
    
    @app.get("/decision/models/summary")
    async def get_decision_models_summary():
        """Get summary of trained Decision ML models."""
        global decision_pipeline
        
        if DECISION_ML_MODE == "fallback":
            return {
                "pipeline_status": "demonstration_mode",
                "mode": "fallback",
                "message": "Running in demonstration mode with basic matching algorithm",
                "data_summary": {
                    "jobs_count": 50,
                    "candidates_count": 5,
                    "feature_count": 10
                }
            }
        
        if decision_pipeline is None:
            return {"message": "No Decision ML pipeline initialized"}
        
        return decision_pipeline.get_pipeline_summary()

else:
    # Placeholder endpoints when Decision ML is not available
    @app.post("/decision/train")
    async def train_decision_model_unavailable():
        raise HTTPException(
            status_code=503, 
            detail="Decision ML components not available. Please install requirements from decision_ml/requirements.txt"
        )
    
    @app.post("/decision/predict")
    async def predict_decision_matches_unavailable():
        raise HTTPException(
            status_code=503, 
            detail="Decision ML components not available. Please install requirements from decision_ml/requirements.txt"
        )

app_handler = app  # Para garantir compatibilidade com Vercel