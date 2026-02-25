"""
Keystroke Dynamics — 1D CNN Classification Web App
====================================================
Backend built with FastAPI and PyTorch.

Endpoints:
  /register  — Register a user with keystroke timing data and retrain the model.
  /predict   — Predict who is typing based on keystroke timing data.
"""

import json
import os
import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# Persistence — save / load dataset to a JSON file
# ---------------------------------------------------------------------------
DATA_FILE = "keystroke_data.json"

def save_dataset():
    """Persist the in-memory dataset to a JSON file."""
    with open(DATA_FILE, "w") as f:
        json.dump(dataset, f)

def load_dataset():
    """Load dataset from disk if the file exists."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

# ---------------------------------------------------------------------------
# App setup — load saved data on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load persisted dataset and retrain model on startup."""
    global dataset
    dataset = load_dataset()
    if dataset:
        print(f"Loaded {len(dataset)} sample(s) from {DATA_FILE}")
        build_and_train_model()
        print("Model retrained from saved data.")
    else:
        print("No saved data found. Starting fresh.")
    yield

app = FastAPI(lifespan=lifespan)

# Serve static files (index.html, app.js)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------
# In-memory dataset & model storage
# ---------------------------------------------------------------------------
# Each entry: {"username": str, "features": [[hold, flight], ...]}
dataset: List[dict] = []
model: Optional[nn.Module] = None
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
# 1D CNN Model (PyTorch)
# ---------------------------------------------------------------------------

class KeystrokeCNN(nn.Module):
    """Lightweight 1D CNN for keystroke-based user classification."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16,
                               kernel_size=3, padding=1)  # same padding
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        # After conv+pool: (16, SEQUENCE_LENGTH//2) → flattened = 16 * 10 = 160
        self.flatten_size = 16 * (SEQUENCE_LENGTH // 2)
        self.fc = nn.Linear(self.flatten_size, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, 2) → transpose to (batch, 2, seq_len) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # raw logits; softmax applied during inference

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

    X_train = np.concatenate(X_list, axis=0)   # (N, 20, 2)
    y_train = np.array(y_list, dtype=np.int64)  # (N,)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)

    # -----------------------------------------------------------------------
    # 1D CNN Architecture (lightweight — trains in seconds)
    # -----------------------------------------------------------------------
    model = KeystrokeCNN(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train quickly (few epochs, small dataset)
    model.train()
    batch_size = 4
    n_samples = X_tensor.size(0)

    for epoch in range(30):
        # Shuffle each epoch
        perm = torch.randperm(n_samples)
        X_shuffled = X_tensor[perm]
        y_shuffled = y_tensor[perm]

        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()

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

    # Save dataset to disk & retrain the model
    save_dataset()
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
    X_tensor = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]  # shape (num_classes,)

    max_prob = float(probabilities.max())
    predicted_idx = int(probabilities.argmax())

    # Reverse lookup: index → username
    idx_to_name = {v: k for k, v in label_map.items()}
    predicted_name = idx_to_name.get(predicted_idx, "Unknown User")

    if max_prob < CONFIDENCE_THRESHOLD:
        return {"match": "Unknown User", "confidence": round(max_prob * 100, 2)}

    return {"match": predicted_name, "confidence": round(max_prob * 100, 2)}
