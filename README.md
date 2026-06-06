# Complaint Auto-Routing System

**ML-powered complaint processing: routing, priority, ETA, similarity — fully offline.**

---

## Architecture

```
User Input (Text / Audio / Video)
        │
        ▼
[Transcription Layer]  ← openai-whisper (local, offline)
  audio/video → text
        │
        ▼
[Inference Engine]
  ├── Priority Classifier     → LogisticRegression + TF-IDF  (High/Medium/Low)
  ├── ETA Regressor           → GradientBoosting + TF-IDF    (days)
  ├── Officer Router          → RandomForest + TF-IDF        (officer_id)
  └── Similarity Search       → TF-IDF cosine (scipy sparse) (top-K)
        │
        ▼
[Streamlit Web App]
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app (models auto-train on first launch)
streamlit run app.py
```

Models train automatically on first run (~30s). You can also retrain manually:

```bash
python scripts/generate_data.py   # regenerate synthetic training data
python scripts/train.py           # retrain all models
```

## Project Structure

```
complaint_system/
├── app.py                      # Streamlit web app (entry point)
├── requirements.txt
├── data/
│   ├── complaints.csv          # Generated training data (800 samples)
│   └── officers.json           # Officer registry
├── models/                     # Saved model artifacts (auto-generated)
│   ├── priority_model.pkl
│   ├── priority_label_encoder.pkl
│   ├── eta_model.pkl
│   ├── officer_model.pkl
│   ├── officer_label_encoder.pkl
│   ├── similarity_vectorizer.pkl
│   ├── similarity_matrix.npz
│   ├── complaint_store.csv
│   └── eval_metrics.json
├── scripts/
│   ├── generate_data.py        # Synthetic dataset generator
│   └── train.py                # Full training pipeline
└── src/
    ├── inference.py            # Inference engine (singleton, lazy-loaded)
    ├── transcription.py        # Audio/video → text (whisper + moviepy)
    └── bootstrap.py            # Auto-train helper for first run
```

## Model Decisions & Tradeoffs

| Task | Model | Reason |
|------|-------|--------|
| Priority (clf) | LogisticRegression + TF-IDF | Fast, interpretable, works well on short text |
| ETA (regression) | GradientBoosting + TF-IDF | Handles non-linear ETA patterns, robust to outliers |
| Officer routing | RandomForest + TF-IDF | Handles imbalanced classes well via `class_weight="balanced"` |
| Similarity | TF-IDF + cosine (scipy sparse) | Zero extra dependencies, scales to 100k+ complaints |
| Transcription | openai-whisper (local) | Multilingual, offline, no API, state-of-art ASR |

**Why not sentence-transformers?**
Sentence-transformers (e.g., `paraphrase-multilingual-MiniLM`) would give better semantic similarity at the cost of ~500MB model download and torch dependency. The current TF-IDF approach achieves Recall@5 = 1.0 on the test set. To upgrade, swap `src/inference.py` similarity section.

## Evaluation Metrics (on synthetic 800-sample dataset)

| Task | Metric | Score |
|------|--------|-------|
| Priority Classification | F1 (weighted) | 1.00 |
| ETA Prediction | MAE | ~1.09 days |
| Officer Routing | F1 (weighted) | 1.00 |
| Similarity Retrieval | Recall@5 | 1.00 |

> Note: These scores are on the synthetic dataset. Real-world performance will depend on actual complaint data quality and distribution.

## Multilingual Support

- **Text**: TF-IDF is language-agnostic. Works on any UTF-8 text (Hindi, Odia, English, etc.)
- **Audio/Video**: Whisper natively handles 99+ languages — pass any language audio

## Extending to Real Data

1. Replace `data/complaints.csv` with real labeled complaint data
2. Ensure columns: `text`, `priority`, `eta_days`, `officer_id`, `officer_name`, `domain`
3. Update `data/officers.json` with real officer list
4. Run `python scripts/train.py`

<img width="1918" height="965" alt="Screenshot from 2026-06-06 09-53-30" src="https://github.com/user-attachments/assets/2576b6f4-f104-4d15-95ba-14b68a790927" />
<img width="1918" height="965" alt="Screenshot from 2026-06-06 09-53-10" src="https://github.com/user-attachments/assets/70640004-a872-4ca9-8ee3-414081a903c6" />


## Audio/Video Setup

```bash
pip install openai-whisper moviepy
# whisper downloads model weights (~75MB for 'base') on first use
```

## No External APIs Used

- No OpenAI API, Gemini, AWS, Google Cloud, or any paid service
- Whisper runs fully locally
- All models are trained and served locally
