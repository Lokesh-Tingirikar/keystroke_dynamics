/**
 * Keystroke Dynamics — Frontend (Vanilla JS)
 * ===========================================
 * Captures hold times and flight times for every keystroke and sends
 * them to the FastAPI backend for registration or prediction.
 *
 * Hold time  = release timestamp − press timestamp   (same key)
 * Flight time = press timestamp of current key − release timestamp of previous key
 */

// Minimum number of keystrokes required before sending data
const MIN_KEYSTROKES = 5;

// ---------------------------------------------------------------------------
// Keystroke recorder class — reusable for any textarea
// ---------------------------------------------------------------------------
class KeystrokeRecorder {
  constructor() {
    this.reset();
  }

  /** Clear all recorded data. */
  reset() {
    this._pressMap = {};      // keyCode → press timestamp
    this._lastRelease = null; // timestamp of the most recent key release
    this.holdTimes = [];
    this.flightTimes = [];
  }

  /** Call on every keydown event. */
  onKeyDown(event) {
    const key = event.key;
    if (!this._pressMap[key]) {
      this._pressMap[key] = performance.now();
    }
  }

  /** Call on every keyup event. */
  onKeyUp(event) {
    const key = event.key;
    const now = performance.now();
    const pressTime = this._pressMap[key];

    if (pressTime) {
      // Hold time (ms): how long the key was held
      const holdTime = now - pressTime;
      this.holdTimes.push(holdTime);

      // Flight time (ms): gap between previous key release and this key press
      if (this._lastRelease !== null) {
        const flightTime = pressTime - this._lastRelease;
        this.flightTimes.push(flightTime);
      } else {
        // First key — no previous release; use 0
        this.flightTimes.push(0);
      }

      this._lastRelease = now;
      delete this._pressMap[key];
    }
  }

  /** Return the recorded data as a JSON-friendly object. */
  getData() {
    return {
      hold_times: this.holdTimes,
      flight_times: this.flightTimes,
    };
  }
}

// ---------------------------------------------------------------------------
// Attach recorders to the two textareas
// ---------------------------------------------------------------------------
const registerRecorder = new KeystrokeRecorder();
const testRecorder = new KeystrokeRecorder();

const registerText = document.getElementById("registerText");
const testText = document.getElementById("testText");

registerText.addEventListener("keydown", (e) => registerRecorder.onKeyDown(e));
registerText.addEventListener("keyup", (e) => registerRecorder.onKeyUp(e));

testText.addEventListener("keydown", (e) => testRecorder.onKeyDown(e));
testText.addEventListener("keyup", (e) => testRecorder.onKeyUp(e));

// ---------------------------------------------------------------------------
// Register & Train
// ---------------------------------------------------------------------------
document.getElementById("registerBtn").addEventListener("click", async () => {
  const username = document.getElementById("username").value.trim();
  const statusEl = document.getElementById("status");

  if (!username) {
    statusEl.textContent = "⚠️  Please enter a user name.";
    return;
  }

  const data = registerRecorder.getData();
  if (data.hold_times.length < MIN_KEYSTROKES) {
    statusEl.textContent = "⚠️  Please type at least a few words so we can capture enough keystrokes.";
    return;
  }

  statusEl.textContent = "⏳ Registering & training…";

  try {
    const response = await fetch("/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, keystrokes: data }),
    });
    const result = await response.json();
    statusEl.textContent = "✅ " + result.message;
  } catch (err) {
    statusEl.textContent = "❌ Error: " + err.message;
  }

  // Reset recorder & textarea for next sample
  registerRecorder.reset();
  registerText.value = "";
});

// ---------------------------------------------------------------------------
// Identify User (Predict)
// ---------------------------------------------------------------------------
document.getElementById("predictBtn").addEventListener("click", async () => {
  const resultEl = document.getElementById("result");
  const data = testRecorder.getData();

  if (data.hold_times.length < MIN_KEYSTROKES) {
    resultEl.textContent = "⚠️  Please type at least a few words before testing.";
    resultEl.className = "warning";
    return;
  }

  resultEl.textContent = "⏳ Identifying…";
  resultEl.className = "";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ keystrokes: data }),
    });
    const result = await response.json();

    if (result.match === "Unknown User") {
      resultEl.textContent = `Different User / Unknown — Confidence: ${result.confidence}%`;
      resultEl.className = "warning";
    } else {
      resultEl.textContent = `Matched: ${result.match} — Confidence: ${result.confidence}%`;
      resultEl.className = "success";
    }
  } catch (err) {
    resultEl.textContent = "❌ Error: " + err.message;
    resultEl.className = "";
  }

  // Reset recorder & textarea for next test
  testRecorder.reset();
  testText.value = "";
});
