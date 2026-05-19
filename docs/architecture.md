# SocialPulse AI — Architecture Documentation

## System Overview

SocialPulse AI is a real-time misinformation and trend analysis platform that aggregates social media data, applies ML-powered analysis, and presents insights through an interactive dashboard.

![Architecture Diagram](./architecture-diagram.png)

---

## High-Level Architecture

The system follows a **layered architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────┐
│                  Frontend Layer                  │
│          Next.js + React + Chart.js              │
│         (Analytics Dashboard & UI)               │
├─────────────────────────────────────────────────┤
│                  API Layer                       │
│            FastAPI REST API                      │
│     (Rate Limiting, Auth, Routing)              │
├─────────────────────────────────────────────────┤
│               Services Layer                     │
│  Sentiment │ Classification │ Trending │ FactCheck│
├─────────────────────────────────────────────────┤
│              ML Models Layer                     │
│  RoBERTa │ DistilBERT+VADER │ BERTopic          │
├─────────────────────────────────────────────────┤
│           Data Processing Layer                  │
│     Preprocessing │ Pipeline │ Language Detection│
├─────────────────────────────────────────────────┤
│           Data Collection Layer                  │
│     Reddit Collector │ Twitter Collector │ Dedup │
├─────────────────────────────────────────────────┤
│            Infrastructure Layer                  │
│        MongoDB │ Redis │ Celery │ Docker         │
└─────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Frontend Layer

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Dashboard | Next.js App Router | Main analytics UI |
| Live Feed | React + WebSocket | Real-time post stream |
| Sentiment Timeline | Chart.js | Sentiment over time charts |
| Trending Topics | React Components | Topic visualization |
| Post Explorer | React Table | Searchable post browser |
| Fake/Real Chart | Chart.js | Classification distribution |

**Key Files:**
- [`frontend/app/layout.tsx`](frontend/app/layout.tsx) — Root layout
- [`frontend/app/page.tsx`](frontend/app/page.tsx) — Main dashboard page
- [`frontend/lib/api.ts`](frontend/lib/api.ts) — API client utilities

**Data Flow:**
- Frontend calls REST API endpoints → receives JSON → renders charts/tables
- Future: WebSocket connection for live updates

---

### 2. API Layer

| Endpoint | Method | Service | Description |
|----------|--------|---------|-------------|
| `/api/v1/trending` | GET | TrendingService | Get trending topics |
| `/api/v1/trending/realtime` | GET | TrendingService | Real-time trending |
| `/api/v1/sentiment/{topic}` | GET | SentimentService | Topic sentiment timeline |
| `/api/v1/sentiment/overall` | GET | SentimentService | Overall sentiment breakdown |
| `/api/v1/classify` | POST | ClassificationService | Classify text (fake/real + sentiment) |
| `/api/v1/feed` | GET | FeedService | Filtered post feed |
| `/api/v1/export` | GET | ExportService | Export data (JSON/CSV) |

**Key Files:**
- [`backend/app/main.py`](backend/app/main.py) — FastAPI app entry point
- [`backend/app/api/router.py`](backend/app/api/router.py) — Route aggregation
- [`backend/app/core/config.py`](backend/app/core/config.py) — Settings & configuration
- [`backend/app/core/security.py`](backend/app/core/security.py) — API key authentication

**Middleware:**
- CORS (allow all origins in development)
- Rate limiting (100 req/min default)
- API key authentication

---

### 3. Services Layer

Services encapsulate business logic and coordinate between the API and ML/data layers.

#### SentimentService
- [`backend/app/services/sentiment_service.py`](backend/app/services/sentiment_service.py)
- Aggregates sentiment data by topic and time window
- Uses MongoDB aggregation pipelines for efficient queries

#### ClassificationService
- [`backend/app/services/classification_service.py`](backend/app/services/classification_service.py)
- Orchestrates fake news detection + sentiment analysis
- Returns combined classification + sentiment result

#### TrendingService
- [`backend/app/services/trending_service.py`](backend/app/services/trending_service.py)
- Queries trends collection and computes real-time trending
- Groups posts by subreddit/hashtag with engagement metrics

#### FactCheckService
- [`backend/app/services/factcheck_service.py`](backend/app/services/factcheck_service.py)
- Delegates to the fact_check verifier module
- Returns verification status with sources

---

### 4. ML Models Layer

#### Fake News Detector (RoBERTa)
- **Model:** Fine-tuned RoBERTa-base for binary classification
- **Files:**
  - [`ml_models/fake_news/train.py`](ml_models/fake_news/train.py) — Training script
  - [`ml_models/fake_news/predict.py`](ml_models/fake_news/predict.py) — Prediction with fallback
- **Fallback:** If saved model unavailable, uses pretrained RoBERTa-base
- **Output:** `{ label: "fake"|"real", confidence: 0-100 }`

#### Sentiment Analyzer (DistilBERT + VADER)
- **Model:** Fine-tuned DistilBERT-base-uncased (3-class: positive/neutral/negative)
- **Fallback:** VADER sentiment analyzer for lightweight processing
- **Files:**
  - [`ml_models/sentiment/train.py`](ml_models/sentiment/train.py) — Training script
  - [`ml_models/sentiment/predict.py`](ml_models/sentiment/predict.py) — Prediction with VADER fallback
- **Output:** `{ label: "positive"|"neutral"|"negative", score: 0-100, model: "distilbert"|"vader" }`

#### Topic Modeler (BERTopic)
- **Model:** BERTopic with dynamic embedding-based clustering
- **Files:**
  - [`ml_models/topic_modeling/train.py`](ml_models/topic_modeling/train.py) — Training script
  - [`ml_models/topic_modeling/predict.py`](ml_models/topic_modeling/predict.py) — Topic prediction
- **Output:** `{ topic: int, keywords: list[str] }` per document

#### Higher-Level Model Wrappers

The `fake_news_detector`, `sentiment_analyzer`, and `topic_modeler` packages provide higher-level interfaces:

- [`ml_models/fake_news_detector/model.py`](ml_models/fake_news_detector/model.py) — Wrapper with batch processing & confidence thresholds
- [`ml_models/sentiment_analyzer/model.py`](ml_models/sentiment_analyzer/model.py) — Wrapper with VADER/DistilBERT switching & aggregation
- [`ml_models/topic_modeler/model.py`](ml_models/topic_modeler/model.py) — Wrapper with topic labeling & trend detection

---

### 5. Data Processing Layer

| Component | File | Purpose |
|-----------|------|---------|
| TextPreprocessor | [`data_processing/preprocessor.py`](data_processing/preprocessor.py) | Clean, tokenize, detect language, VADER sentiment |
| Pipeline | [`data_processing/pipeline.py`](data_processing/pipeline.py) | Batch/single post processing orchestration |
| Language Detection | [`data_processing/language_detection.py`](data_processing/language_detection.py) | Advanced language detection & filtering |

**Preprocessing Steps:**
1. URL removal
2. Hashtag expansion (#word → word)
3. @mention removal
4. HTML tag stripping
5. Special character removal
6. Whitespace normalization
7. Language detection (langdetect + spaCy)
8. Tokenization & lemmatization (spaCy)
9. VADER sentiment scoring

---

### 6. Data Collection Layer

| Component | File | Purpose |
|-----------|------|---------|
| Reddit Collector | [`data_collection/reddit_collector.py`](data_collection/reddit_collector.py) | PRAW-based Reddit data fetching |
| Twitter Collector | [`data_collection/twitter_collector.py`](data_collection/twitter_collector.py) | Tweepy-based X/Twitter data fetching |
| Deduplication | [`data_collection/deduplication.py`](data_collection/deduplication.py) | Duplicate post detection & removal |
| Scheduler | [`data_collection/scheduler.py`](data_collection/scheduler.py) | 15-minute polling scheduler |

**Collection Flow:**
```
Scheduler (15min) → Collector → Raw Posts DB → Deduplication → Processing Pipeline
```

---

### 7. Fact Check Layer

| Component | File | Purpose |
|-----------|------|---------|
| Google Fact Check | [`fact_check/google_factcheck.py`](fact_check/google_factcheck.py) | Google Fact Check Tools API |
| ClaimBuster | [`fact_check/claimbuster.py`](fact_check/claimbuster.py) | ClaimBuster API for claim detection |
| Verifier | [`fact_check/verifier.py`](fact_check/verifier.py) | Unified verification orchestrator |

---

### 8. Infrastructure Layer

| Service | Technology | Purpose |
|---------|-----------|---------|
| Database | MongoDB (Motor async driver) | Post storage, trends, sentiment records |
| Cache/Queue | Redis | Celery task queue, caching |
| Task Worker | Celery | Async data collection & processing |
| Containerization | Docker + Docker Compose | MongoDB + Redis infrastructure |

**Database Collections:**
- `raw_posts` — Unprocessed social media posts
- `cleaned_posts` — Preprocessed and analyzed posts
- `trends` — Topic trend records
- `fact_checks` — Fact-check verification results
- `sentiment` — Sentiment analysis records

**Fallback Mode:**
When MongoDB is unavailable, the system uses an **in-memory store** with cursor-like query support (defined in [`backend/app/core/database.py`](backend/app/core/database.py)). This enables development and testing without requiring a MongoDB instance.

---

## Data Flow

### Complete Processing Pipeline

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│  Reddit API   │────▶│ Raw Posts DB  │────▶│ Deduplication │
│  Twitter API  │────▶│               │     │              │
└──────────────┘     └───────────────┘     └──────────────┘
                                                   │
                                                   ▼
                                          ┌──────────────┐
                                          │ Preprocessing │
                                          │  Pipeline     │
                                          └──────────────┘
                                                   │
                              ┌────────────────────┼────────────────────┐
                              ▼                    ▼                    ▼
                     ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
                     │ Sentiment    │    │ Fake News    │    │ Topic        │
                     │ Analysis     │    │ Detection    │    │ Modeling     │
                     └──────────────┘    └──────────────┘    └──────────────┘
                              │                    │                    │
                              ▼                    ▼                    ▼
                     ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
                     │ Sentiment DB │    │ Classification│   │ Trends DB    │
                     └──────────────┘    └──────────────┘    └──────────────┘
                              │                    │                    │
                              └────────────────────┼────────────────────┘
                                                   ▼
                                          ┌──────────────┐
                                          │  REST API    │
                                          │  (FastAPI)   │
                                          └──────────────┘
                                                   │
                                                   ▼
                                          ┌──────────────┐
                                          │  Dashboard   │
                                          │  (Next.js)   │
                                          └──────────────┘
```

---

## Security

- **API Key Authentication:** All API endpoints require an API key (configured via `API_KEY_SECRET`)
- **Rate Limiting:** 100 requests/minute default (configurable via `API_RATE_LIMIT`)
- **CORS:** Configurable origins (currently `*` for development)
- **Environment Variables:** All secrets stored in `.env`, never hardcoded
- **Input Validation:** Pydantic schemas validate all request/response data

---

## Scalability Considerations

### Current Design
- Single-instance FastAPI server
- MongoDB for persistent storage
- In-memory fallback for development

### Future Scaling Paths
- **Horizontal scaling:** Multiple FastAPI workers behind a load balancer
- **MongoDB sharding:** Shard by platform or date for large datasets
- **Redis caching:** Cache frequent query results (trending, overall sentiment)
- **Celery workers:** Scale data collection workers independently
- **Model serving:** Deploy ML models as separate microservices (e.g., TorchServe)

---

## Deployment

### Development
```bash
docker-compose up -d          # MongoDB + Redis
uvicorn app.main:app --reload  # Backend
npm run dev                    # Frontend
```

### Production
See [`deployment/docker-compose.prod.yml`](deployment/docker-compose.prod.yml) for production Docker configuration.

| Container | Image | Purpose |
|-----------|-------|---------|
| backend | [`deployment/Dockerfile.backend`](deployment/Dockerfile.backend) | FastAPI server |
| frontend | [`deployment/Dockerfile.frontend`](deployment/Dockerfile.frontend) | Next.js server |
| mongodb | mongo:latest | Database |
| redis | redis:latest | Cache & task queue |

---

## Configuration Reference

All configuration is managed through [`backend/app/core/config.py`](backend/app/core/config.py) with environment variable overrides:

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | development | Environment mode |
| `DEBUG` | True | Debug mode toggle |
| `API_RATE_LIMIT` | 100 | Requests per minute limit |
| `API_KEY_SECRET` | change_this... | API authentication key |
| `MONGODB_URI` | mongodb://localhost:27017 | MongoDB connection string |
| `MONGODB_DB_NAME` | socialpulse_ai | Database name |
| `REDIS_URL` | redis://localhost:6379/0 | Redis connection string |
| `REDDIT_CLIENT_ID` | "" | Reddit API client ID |
| `REDDIT_CLIENT_SECRET` | "" | Reddit API client secret |
| `X_API_KEY` | "" | Twitter/X API key |
| `X_BEARER_TOKEN` | "" | Twitter/X bearer token |
| `GOOGLE_FACT_CHECK_API_KEY` | "" | Google Fact Check API key |
| `CLAIMBUSTER_API_KEY` | "" | ClaimBuster API key |