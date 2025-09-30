# Decision ML - Advanced Recruitment Matching System

## Visão Geral do Projeto

### Objetivo
O Decision ML é uma solução avançada de Machine Learning desenvolvida para resolver os desafios de recrutamento e seleção da empresa Decision. O sistema utiliza algoritmos de aprendizado de máquina para automatizar e otimizar o processo de matching entre vagas e candidatos, com foco especial em vagas de tecnologia de nível júnior.

### Solução Proposta
Construção de uma pipeline completa de machine learning que inclui:
- Pré-processamento avançado de dados de vagas e candidatos
- Engenharia de features especializada para recrutamento
- Múltiplos modelos de ML com seleção automática do melhor
- API REST para deployment em produção
- Sistema de monitoramento contínuo com detecção de drift
- Testes unitários abrangentes
- Containerização com Docker

### Stack Tecnológica
- **Linguagem**: Python 3.11
- **Frameworks de ML**: scikit-learn, pandas, numpy
- **API**: FastAPI
- **Serialização**: joblib
- **Testes**: pytest, pytest-cov
- **Empacotamento**: Docker
- **Deploy**: Local/Cloud (Heroku, AWS, GCP, etc.)
- **Monitoramento**: logging + dashboard de drift personalizado

## Estrutura do Projeto

```
decision_ml/
├── __init__.py                 # Inicialização do pacote
├── api.py                     # API FastAPI para deployment
├── data_preprocessing.py      # Pré-processamento de dados
├── feature_engineering.py     # Engenharia de features
├── model_training.py          # Treinamento de modelos ML
├── pipeline.py               # Pipeline principal
├── monitoring.py             # Sistema de monitoramento
├── requirements.txt          # Dependências Python
├── models/                   # Modelos treinados salvos
├── logs/                     # Logs do sistema
└── tests/                    # Testes unitários
    ├── __init__.py
    └── test_pipeline.py

# Arquivos de configuração
├── Dockerfile                # Configuração Docker
├── docker-compose.yml        # Orquestração de containers
├── train_decision_model.py   # Script de treinamento
└── DECISION_ML_README.md     # Esta documentação
```

## Instruções de Deploy

### Pré-requisitos
- Python 3.11+
- Docker (opcional, mas recomendado)
- 4GB+ RAM disponível
- Arquivos de dados em `JSONs_DECISION/`

### Instalação Local

1. **Clone o repositório e navegue para o diretório**:
```bash
git clone <url-do-repositorio>
cd <nome-da-pasta>
```

2. **Instale as dependências**:
```bash
pip install -r decision_ml/requirements.txt
```

3. **Baixe dados NLTK necessários**:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

4. **Treine o modelo**:
```bash
python train_decision_model.py
```

5. **Execute a API**:
```bash
python -m uvicorn decision_ml.api:app --host 0.0.0.0 --port 8000
```

### Deploy com Docker

1. **Build da imagem**:
```bash
docker build -t decision-ml .
```

2. **Execute o container**:
```bash
docker run -p 8000:8000 -v $(pwd)/decision_ml/models:/app/decision_ml/models decision-ml
```

3. **Ou use docker-compose**:
```bash
docker-compose up -d
```

### Deploy na Nuvem

#### Heroku
```bash
# Instalar Heroku CLI
heroku create decision-ml-app
heroku container:push web
heroku container:release web
```

#### AWS ECS/Fargate
```bash
# Build e push para ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker build -t decision-ml .
docker tag decision-ml:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/decision-ml:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/decision-ml:latest
```

#### Google Cloud Run
```bash
gcloud builds submit --tag gcr.io/PROJECT-ID/decision-ml
gcloud run deploy --image gcr.io/PROJECT-ID/decision-ml --platform managed
```

## Exemplos de Chamadas à API

### 1. Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

**Resposta**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "components": {
    "decision_ml": "available",
    "decision_pipeline_trained": true
  }
}
```

### 2. Treinar Modelo
```bash
curl -X POST "http://localhost:8000/decision/train" \
  -H "Content-Type: application/json" \
  -d '{
    "jobs_path": "JSONs_DECISION/vagas_padrao.json",
    "candidates_path": "JSONs_DECISION/candidates.json"
  }'
```

**Resposta**:
```json
{
  "message": "Decision ML model training started",
  "status": "training",
  "jobs_path": "JSONs_DECISION/vagas_padrao.json",
  "candidates_path": "JSONs_DECISION/candidates.json"
}
```

### 3. Verificar Status do Treinamento
```bash
curl -X GET "http://localhost:8000/decision/status"
```

**Resposta**:
```json
{
  "status": "completed",
  "message": "Decision ML model training completed successfully",
  "timestamp": "2024-01-15T10:45:00",
  "best_model": "random_forest",
  "best_score": 0.8542,
  "pipeline_summary": {
    "jobs_count": 150,
    "candidates_count": 89,
    "job_candidate_pairs": 13350
  }
}
```

### 4. Fazer Predição
```bash
curl -X POST "http://localhost:8000/decision/predict" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "job_description=Desenvolvedor Python com experiência em Django e SQL&top_k=3"
```

**Resposta**:
```json
{
  "vaga": "Desenvolvedor Python com experiência em Django e SQL",
  "top_candidatos": [
    {
      "candidato": "João Silva",
      "score": 0.892,
      "detailed_scores": {
        "overall_score": 85.4,
        "skill_match": 92.3,
        "experience_compatibility": 100.0,
        "education_compatibility": 100.0,
        "language_compatibility": 75.0,
        "text_similarity": 78.9
      }
    },
    {
      "candidato": "Maria Santos",
      "score": 0.834,
      "detailed_scores": {
        "overall_score": 79.2,
        "skill_match": 87.5,
        "experience_compatibility": 80.0,
        "education_compatibility": 100.0,
        "language_compatibility": 85.0,
        "text_similarity": 72.1
      }
    }
  ],
  "model_used": "random_forest",
  "response_time_ms": 245.7,
  "timestamp": "2024-01-15T11:00:00"
}
```

### 5. Monitoramento
```bash
# Health do sistema de monitoramento
curl -X GET "http://localhost:8000/decision/monitoring/health"

# Relatório completo de monitoramento
curl -X GET "http://localhost:8000/decision/monitoring/report"
```

### 6. Usando Python
```python
import requests

# Fazer predição
response = requests.post(
    "http://localhost:8000/decision/predict",
    data={
        "job_description": "Engenheiro de Dados com Python e SQL",
        "top_k": 5
    }
)

predictions = response.json()
print(f"Top candidatos: {predictions['top_candidatos']}")
```

## Etapas do Pipeline de Machine Learning

### 1. Pré-processamento dos Dados
- **Limpeza de texto**: Remoção de caracteres especiais, normalização
- **Extração de habilidades**: Identificação automática de skills técnicas
- **Normalização de níveis**: Padronização de experiência e educação
- **Tratamento de dados faltantes**: Estratégias específicas por tipo de campo

### 2. Engenharia de Features
- **Matching de habilidades**: Score ponderado baseado em skills técnicas
- **Compatibilidade de experiência**: Algoritmo de matching por nível
- **Compatibilidade educacional**: Score baseado em requisitos mínimos
- **Similaridade semântica**: TF-IDF + cosine similarity
- **Features de ranking**: Percentis relativos por vaga

### 3. Treinamento e Validação
- **Múltiplos algoritmos**: Random Forest, Gradient Boosting, SVM, Logistic Regression
- **Hyperparameter tuning**: Grid Search com validação cruzada
- **Validação estratificada**: Preservação da distribuição de classes
- **Métricas de avaliação**: ROC-AUC, Accuracy, Precision, Recall

### 4. Seleção de Modelo
- **Critério principal**: ROC-AUC score
- **Validação cruzada**: 5-fold cross-validation
- **Teste de generalização**: Hold-out test set
- **Análise de feature importance**: Interpretabilidade do modelo

### 5. Pós-processamento
- **Calibração de probabilidades**: Ajuste de scores de confiança
- **Ranking personalizado**: Ordenação otimizada por vaga
- **Filtragem de resultados**: Aplicação de thresholds mínimos

## Métricas de Avaliação

### Métricas Principais
- **ROC-AUC**: Área sob a curva ROC (métrica principal para seleção de modelo)
- **Accuracy**: Precisão geral do modelo
- **Precision**: Precisão para matches positivos
- **Recall**: Cobertura de matches verdadeiros

### Por que ROC-AUC?
O ROC-AUC foi escolhido como métrica principal porque:
1. **Robustez a desbalanceamento**: Funciona bem mesmo com classes desbalanceadas
2. **Interpretabilidade**: Representa a probabilidade de ranquear um match positivo acima de um negativo
3. **Threshold-independent**: Não depende de um threshold específico
4. **Comparabilidade**: Permite comparação direta entre diferentes modelos

### Confiabilidade para Produção
O modelo é considerado confiável para produção quando:
- ROC-AUC > 0.75 (boa capacidade discriminativa)
- Validação cruzada estável (std < 0.05)
- Performance consistente no test set
- Ausência de overfitting (gap train/validation < 0.1)

## Sistema de Monitoramento

### Funcionalidades
- **Logging estruturado**: Todas as operações são logadas
- **Detecção de drift**: Monitoramento automático de mudanças nos dados
- **Métricas de performance**: Tracking contínuo da qualidade do modelo
- **Alertas automáticos**: Notificações quando thresholds são ultrapassados
- **Relatórios periódicos**: Dashboards de saúde do sistema

### Métricas Monitoradas
- Tempo de resposta das predições
- Distribuição de scores de matching
- Taxa de erro da API
- Uso de recursos do sistema
- Qualidade das predições

## Testes Unitários

### Executar Testes
```bash
# Executar todos os testes
pytest decision_ml/tests/ -v

# Executar com cobertura
pytest decision_ml/tests/ --cov=decision_ml --cov-report=html

# Executar testes específicos
pytest decision_ml/tests/test_pipeline.py::TestDecisionDataPreprocessor -v
```

### Cobertura de Testes
O projeto mantém >80% de cobertura de testes unitários, incluindo:
- Pré-processamento de dados
- Engenharia de features
- Treinamento de modelos
- Pipeline completo
- Casos de erro e edge cases

## Troubleshooting

### Problemas Comuns

1. **Erro de memória durante treinamento**:
   - Reduza o tamanho do dataset ou use sampling
   - Aumente a memória disponível
   - Use processamento em batches

2. **API não responde**:
   - Verifique se o modelo foi treinado
   - Confirme que as dependências estão instaladas
   - Verifique logs em `decision_ml/logs/`

3. **Baixa performance do modelo**:
   - Verifique qualidade dos dados de entrada
   - Ajuste hyperparâmetros
   - Considere feature engineering adicional

4. **Drift detectado**:
   - Analise mudanças nos dados de entrada
   - Considere retreinamento do modelo
   - Ajuste thresholds de detecção

### Logs e Debugging
```bash
# Ver logs em tempo real
tail -f decision_ml/logs/decision_ml_$(date +%Y%m%d).log

# Verificar saúde do sistema
curl http://localhost:8000/decision/monitoring/health

# Debug mode
export LOG_LEVEL=DEBUG
python -m uvicorn decision_ml.api:app --reload
```

## Contribuição

### Desenvolvimento
1. Fork o repositório
2. Crie uma branch para sua feature
3. Implemente mudanças com testes
4. Execute a suite de testes
5. Submeta um Pull Request

### Padrões de Código
- Siga PEP 8 para Python
- Documente funções e classes
- Mantenha cobertura de testes >80%
- Use type hints quando possível

## Licença

Este projeto é desenvolvido para fins educacionais como parte do Datathon da Decision.

---

**Desenvolvido com ❤️ para otimizar processos de recrutamento através de Inteligência Artificial**