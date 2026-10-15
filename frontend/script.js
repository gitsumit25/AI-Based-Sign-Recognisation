// SIGNOVA real-time frontend. The model recognizes only its three trained classes.
const API = ""; // Same origin: the FastAPI server serves this page and the API.
const video = document.getElementById("video");
const placeholder = document.getElementById("cameraPlaceholder");
const statusText = document.getElementById("cameraStatus");
const videoUpload = document.getElementById("videoUpload");

let stream = null;
let recognizing = false;
let recognitionTimer = null;
let detectionCount = 0;
let lastHistorySign = "";
let lastHistoryTime = 0;
const sessionId = (window.crypto && crypto.randomUUID)
    ? crypto.randomUUID()
    : `signova-${Date.now()}-${Math.random().toString(16).slice(2)}`;

function setStatus(text, color = "#64748b") {
    statusText.textContent = text;
    statusText.style.color = color;
}

async function startCamera() {
    if (stream) return true;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Camera access is unavailable in this browser. Open SIGNOVA using http://127.0.0.1:8001.");
        return false;
    }
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
            audio: false
        });
        video.srcObject = stream;
        video.muted = true;
        video.playsInline = true;
        await video.play();
        video.style.display = "block";
        placeholder.style.display = "none";
        setStatus("● Camera Active", "#15803d");
        return true;
    } catch (error) {
        console.error(error);
        setStatus("● Camera unavailable", "#b91c1c");
        alert("Camera access failed. Allow camera permission in your browser and try again.");
        return false;
    }
}

async function startRealtimeRecognition() {
    if (recognizing) return;
    const cameraReady = stream ? true : await startCamera();
    if (!cameraReady) return;

    try {
        await fetch(`${API}/api/recognize-reset`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId })
        });
    } catch (error) {
        console.warn("Could not reset prior session", error);
    }

    recognizing = true;
    lastHistorySign = "";
    setStatus("● Collecting sign frames…", "#b45309");
    showResult("COLLECTING", "0 / 45 frames", "—");
    recognitionLoop();
}

// Kept as an alias in case an older copy of the HTML calls captureGesture().
function captureGesture() {
    return startRealtimeRecognition();
}

async function recognitionLoop() {
    if (!recognizing || !stream) return;
    try {
        if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0) {
            const canvas = document.createElement("canvas");
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const context = canvas.getContext("2d", { willReadFrequently: false });
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            const image = canvas.toDataURL("image/jpeg", 0.72);
            const response = await fetch(`${API}/api/recognize-frame`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: sessionId, image })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || data.error || "Frame recognition failed");

            if (!data.ready) {
                const count = Number(data.frames_collected || 0);
                showResult(count >= 45 ? "CHECK CAMERA" : "COLLECTING", `${count} / 45 frames`, data.message || "Keep both hands and your upper body visible.");
                setStatus(count >= 45 ? "● Waiting for visible landmarks…" : "● Collecting sign frames…", "#b45309");
            } else {
                const result = {
                    sign: data.sign,
                    text: data.text,
                    confidence: Number(data.confidence),
                    scores: data.scores || {}
                };
                showResult(result.sign.toUpperCase(), result.text, `${result.confidence.toFixed(2)}%`);
                setStatus("● Recognition active", "#15803d");
                const now = Date.now();
                if (result.sign !== lastHistorySign || now - lastHistoryTime > 5000) {
                    addHistory(result);
                    lastHistorySign = result.sign;
                    lastHistoryTime = now;
                }
            }
        }
    } catch (error) {
        console.error(error);
        setStatus("● Recognition error", "#b91c1c");
        showResult("ERROR", "Check the SIGNOVA server terminal.", "—");
        recognizing = false;
        return;
    }
    if (recognizing) recognitionTimer = window.setTimeout(recognitionLoop, 70);
}

async function stopRealtimeRecognition() {
    recognizing = false;
    if (recognitionTimer) window.clearTimeout(recognitionTimer);
    recognitionTimer = null;
    try {
        await fetch(`${API}/api/recognize-reset`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId })
        });
    } catch (error) {
        console.warn("Session reset failed", error);
    }
    setStatus(stream ? "● Camera Active" : "● Camera Off", stream ? "#15803d" : "#64748b");
}

async function stopCamera() {
    await stopRealtimeRecognition();
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null;
    video.srcObject = null;
    video.style.display = "none";
    placeholder.style.display = "flex";
    setStatus("● Camera Off", "#64748b");
}

videoUpload.addEventListener("change", async event => {
    const file = event.target.files[0];
    if (!file) return;
    await stopRealtimeRecognition();
    setStatus("● Processing uploaded video…", "#b45309");
    showResult("PROCESSING", "Extracting 45 frames and landmarks…", "—");
    const form = new FormData();
    form.append("file", file, file.name);
    try {
        const response = await fetch(`${API}/api/predict`, { method: "POST", body: form });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || data.error || "Video prediction failed");
        showResult(data.sign.toUpperCase(), data.text, `${Number(data.confidence).toFixed(2)}%`);
        addHistory({ sign: data.sign, text: data.text, confidence: Number(data.confidence) });
        setStatus("● Video processed", "#15803d");
    } catch (error) {
        console.error(error);
        showResult("ERROR", error.message || "Could not process the video.", "—");
        setStatus("● Processing failed", "#b91c1c");
        alert(`SIGNOVA could not process this video.\n\n${error.message}`);
    } finally {
        event.target.value = "";
    }
});

function showResult(sign, text, confidence) {
    document.getElementById("detectedSign").textContent = sign;
    document.getElementById("outputText").textContent = text;
    document.getElementById("confidence").textContent = confidence;
}

function addHistory(result) {
    const table = document.getElementById("history");
    const row = document.createElement("tr");
    detectionCount += 1;
    [String(detectionCount), "🤟", result.text, `${Number(result.confidence).toFixed(2)}%`].forEach(value => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.appendChild(cell);
    });
    table.prepend(row);
}

function clearHistory() {
    document.getElementById("history").innerHTML = "";
    detectionCount = 0;
    lastHistorySign = "";
    lastHistoryTime = 0;
}

function speakResult() {
    const text = document.getElementById("outputText").innerText.trim();
    if (!text || /start the camera|collecting|processing|error|waiting for/i.test(text)) {
        alert("No recognized result is available to speak yet.");
        return;
    }
    if (!("speechSynthesis" in window)) {
        alert("Text-to-speech is not supported by this browser.");
        return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US";
    utterance.rate = 0.9;
    window.speechSynthesis.speak(utterance);
}

function scrollToRecognition() {
    document.getElementById("recognition").scrollIntoView({ behavior: "smooth" });
}

window.startCamera = startCamera;
window.startRealtimeRecognition = startRealtimeRecognition;
window.stopRealtimeRecognition = stopRealtimeRecognition;
window.stopCamera = stopCamera;
window.captureGesture = captureGesture;
window.speakResult = speakResult;
window.clearHistory = clearHistory;
window.scrollToRecognition = scrollToRecognition;
