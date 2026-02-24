"""
Keystroke Dynamics — 1D CNN Classification Web App
====================================================
Backend built with FastAPI and TensorFlow/Keras.

Endpoints:
  /register  — Register a user with keystroke timing data and retrain the model.
  /predict   — Predict who is typing based on keystroke timing data.
"""

import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional

# TensorFlow / Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI()

# Serve static files (index.html, app.js)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------
# In-memory dataset & model storage
# ---------------------------------------------------------------------------
# Each entry: {"username": str, "features": [[hold, flight], ...]}
dataset: List[dict] = []
model: Optional[keras.Model] = None
label_map: dict = {}        # {username: int}
SEQUENCE_LENGTH = 20         # Fixed sequence length for the 1D CNN
CONFIDENCE_THRESHOLD = 0.65  # Below this → "Unknown User"

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class KeystrokeData(BaseModel):
    hold_times: List[float]
    flight_times: List[float]


class RegisterRequest(BaseModel):
    username: str
    keystrokes: KeystrokeData


class PredictRequest(BaseModel):
    keystrokes: KeystrokeData

# ---------------------------------------------------------------------------
# Helper: convert raw keystroke lists → padded numpy array of shape
#         (1, SEQUENCE_LENGTH, 2)
# ---------------------------------------------------------------------------

def preprocess(keystrokes: KeystrokeData) -> np.ndarray:
    """Pad/truncate keystroke features to (SEQUENCE_LENGTH, 2)."""
    hold = keystrokes.hold_times
    flight = keystrokes.flight_times

    # Build pairs [hold_time, flight_time] for each keystroke
    length = min(len(hold), len(flight))
    features = [[hold[i], flight[i]] for i in range(length)]

    # Pad or truncate to SEQUENCE_LENGTH
    if len(features) < SEQUENCE_LENGTH:
        features += [[0.0, 0.0]] * (SEQUENCE_LENGTH - len(features))
    else:
        features = features[:SEQUENCE_LENGTH]

    return np.array(features, dtype=np.float32).reshape(1, SEQUENCE_LENGTH, 2)

# ---------------------------------------------------------------------------
# Helper: build & train a lightweight 1D CNN on the full in-memory dataset
# ---------------------------------------------------------------------------

def build_and_train_model():
    """Build a 1D CNN and train on all registered keystroke samples."""
    global model, label_map

    # Build label map from unique usernames
    usernames = sorted(set(entry["username"] for entry in dataset))
    label_map = {name: idx for idx, name in enumerate(usernames)}
    num_classes = len(usernames)

    # Prepare training arrays
    X_list, y_list = [], []
    for entry in dataset:
        X_list.append(preprocess(
            KeystrokeData(
                hold_times=entry["hold_times"],
                flight_times=entry["flight_times"],
            )
        ))
        y_list.append(label_map[entry["username"]])

    X_train = np.concatenate(X_list, axis=0)                       # (N, 20, 2)
    y_train = keras.utils.to_categorical(y_list, num_classes)       # (N, C)

    # -----------------------------------------------------------------------
    # 1D CNN Architecture (lightweight — trains in seconds)
    # -----------------------------------------------------------------------
    model = keras.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, 2)),
        layers.Conv1D(filters=16, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Train quickly (few epochs, small dataset)
    model.fit(X_train, y_train, epochs=30, batch_size=4, verbose=0)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_index():
    """Serve the main HTML page."""
    return FileResponse("static/index.html")


@app.post("/register")
async def register(req: RegisterRequest):
    """
    Register a new typing sample.
    Appends data to the global dataset and retrains the 1D CNN.
    """
    # Basic username validation
    if not req.username or len(req.username) > 50:
        return {"status": "error", "message": "Username must be 1–50 characters."}

    # Store the raw timing lists together with the username
    dataset.append({
        "username": req.username,
        "hold_times": req.keystrokes.hold_times,
        "flight_times": req.keystrokes.flight_times,
    })

    # Retrain the model on the full dataset
    build_and_train_model()

    return {
        "status": "ok",
        "message": f"User '{req.username}' registered. Model trained on {len(dataset)} sample(s).",
    }


@app.post("/predict")
async def predict(req: PredictRequest):
    """
    Predict the user from a typing sample using the trained 1D CNN.
    Returns 'Unknown User' when the model's confidence is below the threshold.
    """
    if model is None or len(label_map) == 0:
        return {"match": "Unknown User", "confidence": 0.0,
                "message": "No model trained yet. Please register users first."}

    X_test = preprocess(req.keystrokes)
    probabilities = model.predict(X_test, verbose=0)[0]  # shape (num_classes,)
    max_prob = float(np.max(probabilities))
    predicted_idx = int(np.argmax(probabilities))

    # Reverse lookup: index → username
    idx_to_name = {v: k for k, v in label_map.items()}
    predicted_name = idx_to_name.get(predicted_idx, "Unknown User")

    if max_prob < CONFIDENCE_THRESHOLD:
        return {"match": "Unknown User", "confidence": round(max_prob * 100, 2)}

    return {"match": predicted_name, "confidence": round(max_prob * 100, 2)}
