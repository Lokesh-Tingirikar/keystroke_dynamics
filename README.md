# Keystroke Dynamics — 1D CNN User Identification

A web application that identifies users based on their typing patterns using a lightweight **1D Convolutional Neural Network (CNN)**. It captures keystroke timing features — **hold times** and **flight times** — to build a unique biometric profile for each registered user.

## How It Works

| Feature | Description |
|---|---|
| **Hold Time** | Duration a key is held down (key release − key press) |
| **Flight Time** | Gap between releasing the previous key and pressing the next key |

1. **Register** — A random prompt sentence is displayed (MonkeyType-style). Type it under a username. The app records your keystroke timings and trains a 1D CNN on all registered samples.
2. **Identify** — A different prompt is shown. Type it out, and the trained model predicts which registered user is typing based on their unique rhythm.

The UI provides:
- **Live character highlighting** — green for correct, red for mistakes, with a blinking cursor
- **Progress bar** — shows characters and words typed vs. total
- **New Prompt button** — click to get a fresh sentence at any time
- Paste is disabled to ensure genuine typing data

If the model's confidence falls below **65%**, the user is classified as **Unknown User**.

## Tech Stack

- **Backend:** FastAPI + Uvicorn
- **ML Model:** PyTorch (1D CNN)
- **Frontend:** Vanilla HTML, CSS, JavaScript
- **Data:** In-memory (resets on server restart)

## Project Structure

```
keystroke_dynamics/
├── main.py              # FastAPI backend + 1D CNN model
├── requirements.txt     # Python dependencies
├── README.md
└── static/
    ├── index.html       # Frontend UI
    └── app.js           # Keystroke capture & API calls
```

## Prerequisites

- Python 3.9+

## Installation & Running

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd keystroke_dynamics
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the server**
   ```bash
   uvicorn main:app --reload
   ```

5. **Open in browser**
   ```
   http://127.0.0.1:8000
   ```

## API Endpoints

### `POST /register`

Register a user with keystroke timing data and retrain the model.

**Request body:**
```json
{
  "username": "alice",
  "keystrokes": {
    "hold_times": [85.2, 112.0, 95.7],
    "flight_times": [0, 130.5, 98.3]
  }
}
```

**Response:**
```json
{
  "status": "ok",
  "message": "User 'alice' registered. Model trained on 1 sample(s)."
}
```

### `POST /predict`

Predict who is typing based on keystroke timing data.

**Request body:**
```json
{
  "keystrokes": {
    "hold_times": [85.2, 112.0, 95.7],
    "flight_times": [0, 130.5, 98.3]
  }
}
```

**Response:**
```json
{
  "match": "alice",
  "confidence": 92.15
}
```

## Model Architecture

```
Input (20, 2) → Conv1D(16, kernel=3, ReLU) → MaxPool1D(2) → Flatten → Dense(softmax)
```

- Sequences are padded/truncated to a fixed length of **20 timesteps**.
- Trained with **Adam** optimizer and **categorical cross-entropy** loss for **30 epochs**.
- The model retrains from scratch on every new registration (suitable for demo-scale data).

## Notes

- All data is stored **in-memory** — it resets when the server restarts.
- Register **multiple samples per user** for better accuracy.
- At least **2 users** must be registered before identification works reliably.
- Minimum **5 keystrokes** are required per sample.
