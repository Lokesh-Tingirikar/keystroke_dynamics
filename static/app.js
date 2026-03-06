// ===== KEYSTROKE RECORDER =====
// We record TWO things while you type:
//   1) hold_time  = how long you press each key (keydown → keyup)
//   2) flight_time = gap between releasing one key and pressing the next

// Storage for registration typing
let regHoldTimes = [];
let regFlightTimes = [];
let regPressStart = {};    // tracks when each key was pressed
let regLastRelease = null; // when the last key was released

// Storage for test typing
let testHoldTimes = [];
let testFlightTimes = [];
let testPressStart = {};
let testLastRelease = null;

// ===== GRAB HTML ELEMENTS =====
const registerText = document.getElementById("registerText");
const testText = document.getElementById("testText");

// ===== RECORD KEYSTROKES FOR REGISTER BOX =====

registerText.addEventListener("keydown", (e) => {
  // When a key is pressed, save the timestamp
  if (!regPressStart[e.key]) {
    regPressStart[e.key] = performance.now();
  }
});

registerText.addEventListener("keyup", (e) => {
  const now = performance.now();
  const pressTime = regPressStart[e.key];
  if (!pressTime) return;

  // Hold time = now - when key was pressed
  regHoldTimes.push(now - pressTime);

  // Flight time = when this key was pressed - when last key was released
  if (regLastRelease !== null) {
    regFlightTimes.push(pressTime - regLastRelease);
  } else {
    regFlightTimes.push(0); // first key, no previous release
  }

  regLastRelease = now;
  delete regPressStart[e.key];
});

// ===== RECORD KEYSTROKES FOR TEST BOX =====

testText.addEventListener("keydown", (e) => {
  if (!testPressStart[e.key]) {
    testPressStart[e.key] = performance.now();
  }
});

testText.addEventListener("keyup", (e) => {
  const now = performance.now();
  const pressTime = testPressStart[e.key];
  if (!pressTime) return;

  testHoldTimes.push(now - pressTime);

  if (testLastRelease !== null) {
    testFlightTimes.push(pressTime - testLastRelease);
  } else {
    testFlightTimes.push(0);
  }

  testLastRelease = now;
  delete testPressStart[e.key];
});

// ===== REGISTER BUTTON =====

document.getElementById("registerBtn").addEventListener("click", async () => {
  const username = document.getElementById("username").value.trim();
  const status = document.getElementById("status");

  if (!username) {
    status.textContent = "Please enter a name.";
    return;
  }
  if (regHoldTimes.length < 5) {
    status.textContent = "Type a bit more first (at least a few words).";
    return;
  }

  status.textContent = "Registering...";

  // Send the typing data to the server
  const res = await fetch("/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      username: username,
      keystrokes: { hold_times: regHoldTimes, flight_times: regFlightTimes }
    }),
  });
  const data = await res.json();
  status.textContent = data.message;

  // Reset for next registration
  regHoldTimes = [];
  regFlightTimes = [];
  regPressStart = {};
  regLastRelease = null;
  registerText.value = "";
});

// ===== IDENTIFY BUTTON =====

document.getElementById("predictBtn").addEventListener("click", async () => {
  const result = document.getElementById("result");

  if (testHoldTimes.length < 5) {
    result.textContent = "Type a bit more first.";
    return;
  }

  result.textContent = "Identifying...";

  // Send the typing data to the server
  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      keystrokes: { hold_times: testHoldTimes, flight_times: testFlightTimes }
    }),
  });
  const data = await res.json();

  if (data.match === "Unknown User") {
    result.textContent = "Unknown User — Confidence: " + data.confidence + "%";
  } else {
    result.textContent = "Matched: " + data.match + " — Confidence: " + data.confidence + "%";
  }

  // Reset for next test
  testHoldTimes = [];
  testFlightTimes = [];
  testPressStart = {};
  testLastRelease = null;
  testText.value = "";
});
