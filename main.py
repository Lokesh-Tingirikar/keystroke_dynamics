"""
Keystroke Dynamics — Backend
Identifies users by HOW they type (not what they type).
"""

import json, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager


# ==================== DATA STORAGE ====================
# We save all typing samples to a JSON file so data survives restarts

DATA_FILE = "keystroke_data.json"
dataset: list[dict] = []       # all registered typing samples
model = None                    # the trained neural network
label_map: dict = {}            # maps username → number (e.g. "Alice" → 0)
SEQ_LEN = 20                    # we use 20 keystrokes per sample
CONFIDENCE_THRESHOLD = 0.65     # below 65% confidence → "Unknown User"


def save_dataset():
    with open(DATA_FILE, "w") as f:
        json.dump(dataset, f)


def load_dataset():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, ValueError):
            return []
    return []


# ==================== FASTAPI APP ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup: load saved data and train model
    global dataset
    dataset = load_dataset()
    if dataset:
        print(f"Loaded {len(dataset)} samples. Training model...")
        train_model()
    else:
        print("No data yet. Register some users first.")
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ==================== REQUEST SCHEMAS ====================
# These define what JSON the frontend sends us

class KeystrokeData(BaseModel):
    hold_times: list[float]      # how long each key was pressed (ms)
    flight_times: list[float]    # gap between key releases and next press (ms)

class RegisterRequest(BaseModel):
    username: str
    keystrokes: KeystrokeData

class PredictRequest(BaseModel):
    keystrokes: KeystrokeData


# ==================== PREPROCESSING ====================
# Convert variable-length typing data → fixed-size array for the neural network

def preprocess(ks: KeystrokeData) -> np.ndarray:
    length = min(len(ks.hold_times), len(ks.flight_times))
    # Pair up: [[hold1, flight1], [hold2, flight2], ...]
    features = [[ks.hold_times[i], ks.flight_times[i]] for i in range(length)]

    # Pad with zeros if too short, or cut off if too long
    if len(features) < SEQ_LEN:
        features += [[0.0, 0.0]] * (SEQ_LEN - len(features))
    else:
        features = features[:SEQ_LEN]

    return np.array(features, dtype=np.float32).reshape(1, SEQ_LEN, 2)


# ==================== THE NEURAL NETWORK ====================
# A simple 1D CNN: looks at patterns in your typing rhythm
#
#   Input: 20 keystrokes × 2 features (hold, flight)
#     → Conv1d: finds patterns in nearby keystrokes
#     → ReLU: activation function
#     → MaxPool: shrinks the data
#     → Linear: outputs one score per user
#     → Softmax: converts scores to probabilities

class KeystrokeCNN(nn.Module):
    def __init__(self, num_users):
        super().__init__()
        self.conv = nn.Conv1d(2, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(16 * (SEQ_LEN // 2), num_users)

    def forward(self, x):
        x = x.permute(0, 2, 1)          # reshape for Conv1d
        x = self.pool(self.relu(self.conv(x)))
        x = x.view(x.size(0), -1)       # flatten
        return self.fc(x)                # raw scores


# ==================== TRAINING ====================
# Called every time someone registers — retrains on ALL data

def train_model():
    global model, label_map

    # Step 1: map each username to a number
    usernames = sorted(set(e["username"] for e in dataset))
    label_map = {name: i for i, name in enumerate(usernames)}

    # Step 2: prepare training data
    X_list, y_list = [], []
    for e in dataset:
        X_list.append(preprocess(KeystrokeData(
            hold_times=e["hold_times"],
            flight_times=e["flight_times"],
        )))
        y_list.append(label_map[e["username"]])

    X = torch.tensor(np.concatenate(X_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.long)

    # Step 3: create model and train for 30 rounds
    model = KeystrokeCNN(len(usernames))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(30):
        perm = torch.randperm(len(y))
        for i in range(0, len(y), 4):  # batch size = 4
            xb = X[perm[i:i+4]]
            yb = y[perm[i:i+4]]
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()
    model.eval()


# ==================== API ROUTES ====================

@app.get("/")
async def home():
    return FileResponse("static/index.html")


@app.post("/register")
async def register(req: RegisterRequest):
    if not req.username or len(req.username) > 50:
        return {"status": "error", "message": "Username must be 1-50 characters."}

    # Save this typing sample
    dataset.append({
        "username": req.username,
        "hold_times": req.keystrokes.hold_times,
        "flight_times": req.keystrokes.flight_times,
    })
    save_dataset()
    train_model()

    return {"status": "ok", "message": f"'{req.username}' registered. {len(dataset)} total samples."}


@app.post("/predict")
async def predict(req: PredictRequest):
    if model is None:
        return {"match": "Unknown User", "confidence": 0.0,
                "message": "No model yet. Register users first."}

    # Run the typing data through the model
    X = torch.tensor(preprocess(req.keystrokes), dtype=torch.float32)
    with torch.no_grad():
        probs = torch.softmax(model(X), dim=1)[0]

    confidence = float(probs.max())
    predicted_idx = int(probs.argmax())
    idx_to_name = {v: k for k, v in label_map.items()}
    name = idx_to_name.get(predicted_idx, "Unknown User")

    if confidence < CONFIDENCE_THRESHOLD:
        return {"match": "Unknown User", "confidence": round(confidence * 100, 2)}
    return {"match": name, "confidence": round(confidence * 100, 2)}
