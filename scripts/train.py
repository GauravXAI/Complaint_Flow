import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, mean_absolute_error
)
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH   = "data/complaints.csv"
MODELS_DIR  = "models"
OFFICERS_PATH = "data/officers.json"


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows")
    return df


def build_tfidf_pipeline(estimator):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=8000,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            min_df=1,
        )),
        ("clf", estimator),
    ])


def train_priority_model(df):
    print("\n--- Training Priority Classifier ---")
    X = df["text"].values
    y = df["priority"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    model = build_tfidf_pipeline(
        LogisticRegression(max_iter=1000, C=5.0, class_weight="balanced", random_state=42)
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"Priority | Accuracy: {acc:.4f} | F1 (weighted): {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(model, f"{MODELS_DIR}/priority_model.pkl")
    joblib.dump(le,    f"{MODELS_DIR}/priority_label_encoder.pkl")
    return {"accuracy": round(acc, 4), "f1_weighted": round(f1, 4)}


def train_eta_model(df):
    print("\n--- Training ETA Regressor ---")
    X = df["text"].values
    y = df["eta_days"].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_tfidf_pipeline(
        GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"ETA | MAE: {mae:.4f} days")

    joblib.dump(model, f"{MODELS_DIR}/eta_model.pkl")
    return {"mae_days": round(mae, 4)}


def train_officer_model(df):
    print("\n--- Training Officer Router ---")
    X = df["text"].values
    y = df["officer_id"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    model = build_tfidf_pipeline(
        RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced",
            random_state=42, n_jobs=-1
        )
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"Officer Router | Accuracy: {acc:.4f} | F1 (weighted): {f1:.4f}")

    joblib.dump(model, f"{MODELS_DIR}/officer_model.pkl")
    joblib.dump(le,    f"{MODELS_DIR}/officer_label_encoder.pkl")
    return {"accuracy": round(acc, 4), "f1_weighted": round(f1, 4)}


def build_similarity_index(df):
    """
    Build a TF-IDF + cosine similarity index for retrieval.
    Fallback when FAISS isn't installed — uses scipy sparse cosine.
    Recall@5 is computed on held-out same-domain pairs.
    """
    print("\n--- Building Similarity Index ---")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import scipy.sparse

    texts = df["text"].values
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)  # sparse (N, vocab)

    # Compute Recall@5: for each sample, top-5 retrieved should include
    # at least one from the same domain.
    domains = df["domain"].values
    # Sample 100 queries for eval
    np.random.seed(42)
    query_idx = np.random.choice(len(texts), min(100, len(texts)), replace=False)
    hits = 0
    for qi in query_idx:
        q_vec = matrix[qi]
        sims = cosine_similarity(q_vec, matrix).flatten()
        sims[qi] = -1  # exclude self
        top5 = np.argsort(sims)[-5:]
        if any(domains[r] == domains[qi] for r in top5):
            hits += 1
    recall_at_5 = hits / len(query_idx)
    print(f"Similarity | Recall@5: {recall_at_5:.4f}")

    # Persist vectorizer + matrix
    joblib.dump(vectorizer, f"{MODELS_DIR}/similarity_vectorizer.pkl")
    # Store dense or sparse matrix
    scipy.sparse.save_npz(f"{MODELS_DIR}/similarity_matrix.npz", matrix)
    # Store complaint metadata for retrieval
    df.to_csv(f"{MODELS_DIR}/complaint_store.csv", index=False)
    return {"recall_at_5": round(recall_at_5, 4)}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_data()

    metrics = {}
    metrics["priority"]   = train_priority_model(df)
    metrics["eta"]        = train_eta_model(df)
    metrics["officer"]    = train_officer_model(df)
    metrics["similarity"] = build_similarity_index(df)

    with open(f"{MODELS_DIR}/eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Training Complete ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nAll models saved to ./{MODELS_DIR}/")


if __name__ == "__main__":
    main()
