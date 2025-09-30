# Sistema de Matchmaking Inteligente - Decision

## Visão Geral

Este projeto implementa uma plataforma web de matchmaking inteligente que utiliza algoritmos de Machine Learning para encontrar os candidatos mais adequados para vagas de emprego. O sistema foi desenvolvido com FastAPI (backend) e HTML/CSS/JavaScript (frontend), sendo otimizado para deployment na plataforma Vercel.

## Arquitetura do Sistema

### Backend (FastAPI)
- **Linguagem**: Python 3.9+
- **Framework**: FastAPI
- **Funcionalidades principais**:
  - API REST para processamento de vagas
  - Algoritmo de matching baseado em análise de texto
  - Servir interface web estática
  - Suporte a CORS para deployment em nuvem

### Frontend
- **Tecnologias**: HTML5, CSS3, JavaScript (Vanilla)
- **Interface responsiva** com design moderno
- **Duas funcionalidades principais**:
  1. Análise de vaga individual (texto livre)
  2. Análise de múltiplas vagas (upload de arquivo JSON)

### Algoritmo de Matching
- **Método**: Análise de cobertura de habilidades (Coverage Score)
- **Processamento**: Tokenização de texto e extração de skills
- **Filtros**: Remove stopwords e tokens pequenos
- **Resultado**: Top 3 candidatos mais compatíveis por vaga

## Estrutura de Arquivos

```
├── app.py                          # Aplicação principal FastAPI
├── index.html                      # Interface web
├── requirements.txt                # Dependências Python
├── vercel.json                     # Configuração de deployment
├── .vercelignore                   # Arquivos ignorados no deployment
├── Matching/                       # Módulos do algoritmo de matching
│   ├── pipeline.py                 # Pipeline principal de matching
│   ├── preparingJobs.py           # Processamento e filtragem de vagas
│   └── scoring.py                 # Algoritmo de pontuação
├── JSONs/
│   └── candidates.json            # Base de dados de candidatos
├── vagas2.json                    # Arquivo de exemplo para testes
├── test_api.py                    # Script de testes da API
└── DEPLOYMENT.md                  # Guia de deployment
```

## Funcionalidades Detalhadas

### 1. Análise de Vaga Individual (`/match_vaga`)
- **Input**: Descrição textual da vaga
- **Processamento**: 
  - Cria objeto temporário de vaga
  - Carrega base de candidatos
  - Aplica algoritmo de matching
- **Output**: Top 3 candidatos com scores de compatibilidade

### 2. Análise de Múltiplas Vagas (`/match_vagas`)
- **Input**: Arquivo JSON com estrutura específica de vagas
- **Filtros aplicados**:
  - Apenas vagas de nível "Júnior" ou "Analista"
  - Campos específicos extraídos (título, localização, atividades, competências)
- **Output**: Top 3 candidatos para cada vaga processada

### 3. Interface Web
- **Design responsivo** adaptável a diferentes dispositivos
- **Feedback visual** com loading states
- **Tratamento de erros** com mensagens informativas
- **Resultados formatados** em JSON legível

## Algoritmo de Matching

### Extração de Skills
```python
def extract_skills(text: str):
    tokens = re.findall(r'\b\w+\b', text.lower())
    return set(t for t in tokens if t not in STOPWORDS and len(t) > 2)
```

### Cálculo de Score
```python
def coverage_score(job_skills: Set[str], cand_skills: Set[str]) -> float:
    if not job_skills:
        return 0.0
    inter = job_skills.intersection(cand_skills)
    return len(inter) / len(job_skills) * 100
```

**Explicação**: O score representa a porcentagem de habilidades da vaga que o candidato possui.

## Estrutura de Dados

### Formato de Candidatos (candidates.json)
```json
[
    {
        "id": "Nome do Candidato",
        "perfil": "Descrição completa das habilidades e experiências",
        "link": "URL do LinkedIn"
    }
]
```

### Formato de Vagas (entrada para /match_vagas)
```json
{
    "id_vaga": {
        "informacoes_basicas": {
            "titulo_vaga": "Título da posição"
        },
        "perfil_vaga": {
            "pais": "Brasil",
            "estado": "São Paulo",
            "cidade": "São Paulo",
            "nivel profissional": "Júnior",
            "nivel_ingles": "Intermediário",
            "nivel_espanhol": "Básico",
            "principais_atividades": "Descrição das atividades",
            "competencia_tecnicas_e_comportamentais": "Skills requeridas"
        }
    }
}
```

## Configuração e Execução

### Pré-requisitos
- Python 3.9 ou superior
- pip (gerenciador de pacotes Python)

### Instalação Local
```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Executar aplicação
uvicorn app:app --reload

# 3. Acessar interface
# Navegador: http://localhost:8000
```

### Testes
```bash
# Executar testes da API
python test_api.py
```

### Deployment na Vercel
1. Conectar repositório Git à Vercel
2. Vercel detecta automaticamente o projeto Python
3. Usa configuração do `vercel.json`
4. Deploy automático a cada push

## Tecnologias e Dependências

### Backend
- **FastAPI**: Framework web moderno e rápido
- **Uvicorn**: Servidor ASGI para FastAPI
- **Pandas**: Manipulação de dados (se necessário)
- **Python-multipart**: Suporte a upload de arquivos

### Frontend
- **HTML5**: Estrutura semântica
- **CSS3**: Estilização moderna com variáveis CSS
- **JavaScript ES6+**: Funcionalidades interativas
- **Fetch API**: Comunicação com backend

## Características Técnicas

### Segurança
- **CORS configurado** para requests cross-origin
- **Validação de entrada** com FastAPI
- **Tratamento de erros** robusto
- **Limpeza de arquivos temporários**

### Performance
- **Algoritmo otimizado** para matching
- **Processamento assíncrono** com FastAPI
- **Interface responsiva** sem frameworks pesados
- **Caching de resultados** (implícito no navegador)

### Escalabilidade
- **Arquitetura modular** facilita manutenção
- **Separação de responsabilidades** (frontend/backend)
- **Deploy serverless** na Vercel
- **Fácil extensão** do algoritmo de matching

## Limitações e Considerações

1. **Filtro de nível profissional**: Apenas "Júnior" e "Analista"
2. **Idioma**: Otimizado para português brasileiro
3. **Algoritmo simples**: Baseado em intersecção de palavras
4. **Base de candidatos**: Fixa no arquivo JSON

## Possíveis Melhorias

1. **Algoritmo mais sofisticado**: NLP com embeddings
2. **Base de dados dinâmica**: PostgreSQL ou MongoDB
3. **Autenticação**: Sistema de login para empresas
4. **Analytics**: Dashboard com métricas de matching
5. **API mais robusta**: Paginação, filtros avançados
6. **Testes automatizados**: Cobertura completa de código

## Contato e Suporte

Este sistema foi desenvolvido como projeto acadêmico demonstrando conceitos de:
- Desenvolvimento web full-stack
- APIs REST com FastAPI
- Algoritmos de matching
- Deploy em nuvem
- Interface responsiva

Para dúvidas técnicas, consulte a documentação do código ou os comentários inline nos arquivos fonte.