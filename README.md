# Keystroke Dynamics

Identifies users by **how** they type, not what they type.

## What It Does

1. You **register** by typing anything — the app records how long you press each key and the gaps between keys.
2. A small **1D CNN** (neural network) learns your typing rhythm.
3. Later, someone types — the model guesses **who** it is based on their rhythm.

If confidence is below 65%, it says "Unknown User".

## Tech Stack

| Layer | Tech | Files |
|---|---|---|
| **Frontend** | HTML + CSS + JS | `static/index.html`, `static/styles.css`, `static/app.js` |
| **Backend** | Python (FastAPI + PyTorch) | `main.py` |
| **Data** | JSON | `keystroke_data.json` |

## Project Structure

```
keystroke_dynamics/
├── main.py              # Backend — API + neural network
├── main.ipynb           # Same code as main.py, but as a notebook for learning
├── requirements.txt     # Python dependencies
├── README.md
├── keystroke_data.json  # Saved typing data (auto-created)
└── static/
    ├── index.html       # The webpage
    ├── styles.css       # Styling
    └── app.js           # Records keystrokes + talks to the backend
```

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server
uvicorn main:app --reload

# 4. Open browser
# http://localhost:8000
```

## API

### `POST /register`

```json
// Request
{ "username": "alice", "keystrokes": { "hold_times": [85, 112, 95], "flight_times": [0, 130, 98] } }

// Response
{ "status": "ok", "message": "'alice' registered. 1 total samples." }
```

### `POST /predict`

```json
// Request
{ "keystrokes": { "hold_times": [85, 112, 95], "flight_times": [0, 130, 98] } }

// Response
{ "match": "alice", "confidence": 92.15 }
```

## Notes

- Data is saved to `keystroke_data.json` and persists across restarts.
- Register **multiple samples per user** for better accuracy.
- At least **2 users** needed before identification works well.
- Minimum **5 keystrokes** required per sample.
