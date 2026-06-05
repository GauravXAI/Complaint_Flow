import os
import json
import joblib
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass, field
from typing import List, Optional


MODELS_DIR = "models"
DATA_DIR   = "data"


@dataclass
class PredictionResult:
    priority: str
    priority_confidence: float
    eta_days: float
    officer_id: str
    officer_name: str
    officer_domain: str
    similar_complaints: List[dict] = field(default_factory=list)
    input_text: str = ""


class InferenceEngine:
    """
    Singleton-style inference engine.
    All models are loaded once and reused.
    """

    def __init__(self):
        self._loaded = False
        self.priority_model  = None
        self.priority_le     = None
        self.eta_model       = None
        self.officer_model   = None
        self.officer_le      = None
        self.sim_vectorizer  = None
        self.sim_matrix      = None
        self.complaint_store = None
        self.officers        = None

    def load(self):
        if self._loaded:
            return
        self.priority_model  = joblib.load(f"{MODELS_DIR}/priority_model.pkl")
        self.priority_le     = joblib.load(f"{MODELS_DIR}/priority_label_encoder.pkl")
        self.eta_model       = joblib.load(f"{MODELS_DIR}/eta_model.pkl")
        self.officer_model   = joblib.load(f"{MODELS_DIR}/officer_model.pkl")
        self.officer_le      = joblib.load(f"{MODELS_DIR}/officer_label_encoder.pkl")
        self.sim_vectorizer  = joblib.load(f"{MODELS_DIR}/similarity_vectorizer.pkl")
        self.sim_matrix      = scipy.sparse.load_npz(f"{MODELS_DIR}/similarity_matrix.npz")
        self.complaint_store = pd.read_csv(f"{MODELS_DIR}/complaint_store.csv")
        with open(f"{DATA_DIR}/officers.json") as f:
            self.officers = {o["id"]: o for o in json.load(f)}
        self._loaded = True

    def models_exist(self) -> bool:
        required = [
            "priority_model.pkl", "priority_label_encoder.pkl",
            "eta_model.pkl",
            "officer_model.pkl", "officer_label_encoder.pkl",
            "similarity_vectorizer.pkl", "similarity_matrix.npz",
            "complaint_store.csv",
        ]
        return all(os.path.exists(f"{MODELS_DIR}/{f}") for f in required)

    def predict(self, text: str, top_k: int = 5) -> PredictionResult:
        self.load()

        # --- Priority ---
        priority_enc = self.priority_model.predict([text])[0]
        priority_proba = self.priority_model.predict_proba([text])[0]
        priority_label = self.priority_le.inverse_transform([priority_enc])[0]
        priority_conf  = float(np.max(priority_proba))

        # --- ETA ---
        eta = float(self.eta_model.predict([text])[0])
        eta = max(1.0, round(eta, 1))

        # --- Officer ---
        officer_enc = self.officer_model.predict([text])[0]
        officer_id  = self.officer_le.inverse_transform([officer_enc])[0]
        officer_info = self.officers.get(officer_id, {})

        # --- Similarity ---
        q_vec = self.sim_vectorizer.transform([text])
        sims  = cosine_similarity(q_vec, self.sim_matrix).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]

        similar = []
        for idx in top_indices:
            row = self.complaint_store.iloc[idx]
            similar.append({
                "complaint_id": row["complaint_id"],
                "text": row["text"],
                "priority": row["priority"],
                "officer_name": row["officer_name"],
                "eta_days": row["eta_days"],
                "similarity_score": round(float(sims[idx]), 4),
            })

        return PredictionResult(
            priority=priority_label,
            priority_confidence=round(priority_conf, 4),
            eta_days=eta,
            officer_id=officer_id,
            officer_name=officer_info.get("name", officer_id),
            officer_domain=officer_info.get("domain", "unknown"),
            similar_complaints=similar,
            input_text=text,
        )


_engine = InferenceEngine()


def get_engine() -> InferenceEngine:
    return _engine
