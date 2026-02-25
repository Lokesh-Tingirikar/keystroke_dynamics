/**
 * Keystroke Dynamics — Frontend (Vanilla JS)
 * ===========================================
 * MonkeyType-style guided typing with keystroke capture.
 * Captures hold times and flight times for every keystroke and sends
 * them to the FastAPI backend for registration or prediction.
 *
 * Hold time   = release timestamp − press timestamp   (same key)
 * Flight time = press timestamp of current key − release timestamp of previous key
 */

// Minimum number of keystrokes required before sending data
const MIN_KEYSTROKES = 5;

// ---------------------------------------------------------------------------
// Sample prompts (MonkeyType-style)
// ---------------------------------------------------------------------------
const PROMPTS = [
  "the quick brown fox jumps over the lazy dog near the river bank",
  "every morning she walks to the park and watches the sunrise quietly",
  "a gentle breeze swept through the open window bringing fresh air inside",
  "he picked up the old book and started reading from where he left off",
  "the city lights flickered in the distance as the night grew darker",
  "she smiled and waved before crossing the street to the coffee shop",
  "typing patterns are unique to each person like a digital fingerprint",
  "the rain started falling just as they reached the shelter near the lake",
  "coding late at night with a cup of coffee is a peaceful experience",
  "the cat sat on the warm keyboard and refused to move for anyone",
  "bright stars filled the sky as they sat around the campfire telling stories",
  "practice makes perfect when it comes to improving your typing speed",
  "music played softly in the background while she finished her homework",
  "the train arrived on time and they quickly boarded the last carriage",
  "he opened the door slowly and peeked inside the dark empty room",
];

function getRandomPrompt(exclude = "") {
  const available = PROMPTS.filter((p) => p !== exclude);
  return available[Math.floor(Math.random() * available.length)];
}

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
// Prompt renderer — handles display & live progress
// ---------------------------------------------------------------------------
class PromptManager {
  constructor({ displayEl, textareaEl, progressEl, barEl, wordCountEl, labelEl }) {
    this.displayEl = displayEl;
    this.textareaEl = textareaEl;
    this.progressEl = progressEl;
    this.barEl = barEl;
    this.wordCountEl = wordCountEl;
    this.labelEl = labelEl;
    this.prompt = "";
    this.completed = false;
  }

  /** Set a new prompt and render it. */
  setPrompt(text) {
    this.prompt = text;
    this.completed = false;
    this.textareaEl.value = "";
    this.textareaEl.disabled = false;
    this._render("");
    this._updateProgress(0);

    const wordCount = text.split(/\s+/).filter(Boolean).length;
    this.labelEl.textContent = `${wordCount} words · ${text.length} chars`;
  }

  /** Call on every input event to update the display. */
  onInput() {
    const typed = this.textareaEl.value;
    this._render(typed);

    // Count correctly typed characters (only continuous correct from start)
    let correctCount = 0;
    for (let i = 0; i < typed.length && i < this.prompt.length; i++) {
      if (typed[i] === this.prompt[i]) correctCount++;
      else break;
    }

    this._updateProgress(typed.length);

    // Check completion: typed matches the full prompt
    if (typed === this.prompt && !this.completed) {
      this.completed = true;
      this.textareaEl.disabled = true;
    }
  }

  /** Render character-by-character highlights. */
  _render(typed) {
    let html = "";
    for (let i = 0; i < this.prompt.length; i++) {
      const ch = this.prompt[i] === " " ? "&nbsp;" : this._escapeHtml(this.prompt[i]);
      if (i < typed.length) {
        const cls = typed[i] === this.prompt[i] ? "correct" : "incorrect";
        html += `<span class="char ${cls}">${ch}</span>`;
      } else if (i === typed.length) {
        html += `<span class="char current">${ch}</span>`;
      } else {
        html += `<span class="char">${ch}</span>`;
      }
    }
    // Show extra typed chars as errors
    for (let i = this.prompt.length; i < typed.length; i++) {
      const ch = typed[i] === " " ? "&nbsp;" : this._escapeHtml(typed[i]);
      html += `<span class="char incorrect">${ch}</span>`;
    }
    this.displayEl.innerHTML = html;
  }

  _updateProgress(typedLen) {
    const total = this.prompt.length;
    const pct = total > 0 ? Math.min((typedLen / total) * 100, 100) : 0;
    this.progressEl.textContent = `${typedLen} / ${total} chars`;
    this.barEl.style.width = `${pct}%`;

    // Word progress
    const typedText = this.textareaEl.value;
    const wordsTyped = typedText.trim() === "" ? 0 : typedText.trim().split(/\s+/).length;
    const totalWords = this.prompt.split(/\s+/).filter(Boolean).length;
    this.wordCountEl.textContent = `${wordsTyped} / ${totalWords} words`;
  }

  _escapeHtml(c) {
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
    return map[c] || c;
  }

  /** Reset after submit. */
  reset(newPrompt) {
    this.setPrompt(newPrompt || getRandomPrompt(this.prompt));
  }
}

// ---------------------------------------------------------------------------
// Initialize recorders & prompt managers
// ---------------------------------------------------------------------------
const registerRecorder = new KeystrokeRecorder();
const testRecorder = new KeystrokeRecorder();

const registerText = document.getElementById("registerText");
const testText = document.getElementById("testText");

const regPrompt = new PromptManager({
  displayEl: document.getElementById("regPromptDisplay"),
  textareaEl: registerText,
  progressEl: document.getElementById("regProgress"),
  barEl: document.getElementById("regProgressBar"),
  wordCountEl: document.getElementById("regWordCount"),
  labelEl: document.getElementById("regPromptLabel"),
});

const testPrompt = new PromptManager({
  displayEl: document.getElementById("testPromptDisplay"),
  textareaEl: testText,
  progressEl: document.getElementById("testProgress"),
  barEl: document.getElementById("testProgressBar"),
  wordCountEl: document.getElementById("testWordCount"),
  labelEl: document.getElementById("testPromptLabel"),
});

// Set initial prompts
regPrompt.setPrompt(getRandomPrompt());
testPrompt.setPrompt(getRandomPrompt(regPrompt.prompt));

// New prompt buttons
document.getElementById("newRegPrompt").addEventListener("click", () => {
  registerRecorder.reset();
  regPrompt.reset();
});

document.getElementById("newTestPrompt").addEventListener("click", () => {
  testRecorder.reset();
  testPrompt.reset();
});

// Click on prompt box → focus textarea
document.getElementById("regPromptDisplay").addEventListener("click", () => registerText.focus());
document.getElementById("testPromptDisplay").addEventListener("click", () => testText.focus());

// ---------------------------------------------------------------------------
// Attach recorders + live prompt updates
// ---------------------------------------------------------------------------
registerText.addEventListener("keydown", (e) => registerRecorder.onKeyDown(e));
registerText.addEventListener("keyup", (e) => registerRecorder.onKeyUp(e));
registerText.addEventListener("input", () => regPrompt.onInput());

testText.addEventListener("keydown", (e) => testRecorder.onKeyDown(e));
testText.addEventListener("keyup", (e) => testRecorder.onKeyUp(e));
testText.addEventListener("input", () => testPrompt.onInput());

// Prevent paste in typing areas
registerText.addEventListener("paste", (e) => e.preventDefault());
testText.addEventListener("paste", (e) => e.preventDefault());

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

  // Reset recorder & prompt for next sample
  registerRecorder.reset();
  regPrompt.reset();
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

  // Reset recorder & prompt for next test
  testRecorder.reset();
  testPrompt.reset();
});
