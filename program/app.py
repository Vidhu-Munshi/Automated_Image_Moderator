import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
import json
import io

# ======== CREATE APP ========
app = FastAPI()

# ======== ADD CORS ========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======== LOAD MODEL + LABELS ========
MODEL_PATH = "emotion_best.keras"
LABELS_PATH = "emotion_labels.json"

print("📦 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(LABELS_PATH, "r") as f:
    labels = json.load(f)["class_names"]
print(f"✅ Model loaded! Labels: {labels}")


# ======== IMAGE PREPROCESS ========
def read_image(file):
    img = Image.open(io.BytesIO(file)).convert("RGB")
    img = img.resize((48, 48))
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# ======== PREDICT ENDPOINT ========
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        x = read_image(contents)
        probs = model.predict(x)[0]

        top_idx = int(np.argmax(probs))
        top_label = labels[top_idx]
        top_prob = float(probs[top_idx])

        # Check if detected label is a weapon/object
        detected_objects = []
        object_keywords = ['weapon', 'gun', 'knife', 'blood', 'violence', 'drugs', 'alcohol', 'cigarette', 'nudity',
                           'gore']

        # If top prediction is an object, add it to detected_objects
        if top_label.lower() in object_keywords:
            detected_objects.append({
                "label": top_label.lower(),
                "confidence": round(top_prob, 4)
            })

        return {
            "status": "success",
            "prediction": top_label,
            "confidence": round(top_prob, 4),
            "all_probs": {labels[i]: float(probs[i]) for i in range(len(labels))},
            "detected_objects": detected_objects
        }

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "reason": str(e)}
        )


# ======== SERVE FRONTEND AT ROOT ========
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emotion & Safety Detector</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', system-ui, sans-serif; min-height: 100vh; background: linear-gradient(135deg, #f5f1e8 0%, #ebe4d8 25%, #e8dfc9 50%, #ebe4d8 75%, #f5f1e8 100%); padding: 20px; color: #2c2416; position: relative; overflow-x: hidden; }
        body::before { content: ''; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 800"><path d="M0,400 Q300,350 600,400 T1200,400 L1200,800 L0,800 Z" fill="%23d4c5a9" opacity="0.3"/><path d="M0,500 Q300,450 600,500 T1200,500 L1200,800 L0,800 Z" fill="%23c9b591" opacity="0.2"/></svg>'); background-size: cover; pointer-events: none; z-index: 0; }
        .container { position: relative; z-index: 1; }
        .container { max-width: 950px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 24px; }
        .header-badge { display: inline-flex; align-items: center; gap: 8px; background: linear-gradient(135deg, #d4a574, #c9945e); padding: 8px 16px; border-radius: 50px; margin-bottom: 12px; box-shadow: 0 4px 20px rgba(212, 165, 116, 0.4), 0 0 40px rgba(212, 165, 116, 0.2); animation: glow 3s ease-in-out infinite; }
        @keyframes glow { 0%, 100% { box-shadow: 0 4px 20px rgba(212, 165, 116, 0.4), 0 0 40px rgba(212, 165, 116, 0.2); } 50% { box-shadow: 0 6px 30px rgba(212, 165, 116, 0.6), 0 0 60px rgba(212, 165, 116, 0.3); } }
        .header-badge span { color: #f5f1e8; font-size: 14px; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.2); }
        h1 { font-size: 38px; margin-bottom: 8px; background: linear-gradient(135deg, #d4a574, #b8925f, #d4a574); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: 900; letter-spacing: 2px; filter: drop-shadow(0 2px 8px rgba(212, 165, 116, 0.4)); }
        .subtitle { color: #8b7355; font-size: 15px; font-weight: 500; }
        .main-grid { display: grid; grid-template-columns: 300px 1fr; gap: 20px; }
        @media (max-width: 768px) { .main-grid { grid-template-columns: 1fr; } }
        .panel { background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(245, 241, 232, 0.85)); backdrop-filter: blur(10px); border-radius: 18px; border: 2px solid rgba(212, 165, 116, 0.3); overflow: hidden; box-shadow: 0 8px 32px rgba(139, 115, 85, 0.15), 0 0 0 1px rgba(255, 255, 255, 0.5); }
        .tabs { display: flex; border-bottom: 2px solid rgba(212, 165, 116, 0.3); background: rgba(245, 241, 232, 0.5); }
        .tab { flex: 1; padding: 14px; background: none; border: none; color: #8b7355; font-size: 14px; font-weight: 600; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 8px; transition: all 0.3s; }
        .tab:hover { background: rgba(212, 165, 116, 0.15); color: #6b5742; }
        .tab.active { background: linear-gradient(135deg, rgba(212, 165, 116, 0.25), rgba(201, 148, 94, 0.2)); color: #6b5742; border-bottom: 3px solid #d4a574; font-weight: 700; }
        .count { background: rgba(239,68,68,0.8); color: #fff; padding: 2px 8px; border-radius: 10px; font-size: 11px; }
        .filter-panel { padding: 12px; max-height: 380px; overflow-y: auto; }
        .filter-btn { width: 100%; display: flex; align-items: center; gap: 10px; padding: 10px 12px; margin-bottom: 8px; border-radius: 10px; border: 1px solid; cursor: pointer; transition: all 0.2s; background: transparent; color: #fff; }
        .filter-btn:hover { transform: translateX(4px); }
        .filter-btn .emoji { font-size: 20px; }
        .filter-btn .info { flex: 1; text-align: left; }
        .filter-btn .name { display: block; color: #2c2416; font-size: 14px; font-weight: 500; text-transform: capitalize; }
        .filter-btn .desc { display: block; color: #8b7355; font-size: 11px; }
        .filter-btn .status { font-size: 16px; }
        .filter-btn .badge-high { background: rgba(239,68,68,0.5); color: #fca5a5; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; }
        .allowed { background: rgba(34,197,94,0.15); border-color: rgba(74,222,128,0.4); }
        .blocked { background: rgba(239,68,68,0.15); border-color: rgba(248,113,113,0.4); }
        .blocked-high { background: rgba(239,68,68,0.2); border-color: rgba(239,68,68,0.5); }
        .blocked-medium { background: rgba(249,115,22,0.2); border-color: rgba(249,115,22,0.5); }
        .blocked-low { background: rgba(234,179,8,0.2); border-color: rgba(234,179,8,0.5); }
        .warning-note { margin: 12px; padding: 12px; background: rgba(212, 165, 116, 0.15); border: 1px solid rgba(212, 165, 116, 0.3); border-radius: 10px; }
        .warning-note p { color: #6b5742; font-size: 12px; }
        .upload-panel { padding: 20px; }
        #dropzone { border: 3px dashed rgba(212, 165, 116, 0.4); border-radius: 16px; padding: 50px 20px; text-align: center; cursor: pointer; transition: all 0.3s; background: rgba(255, 255, 255, 0.5); }
        #dropzone:hover, #dropzone.drag-over { border-color: #d4a574; background: rgba(212, 165, 116, 0.1); box-shadow: 0 4px 20px rgba(212, 165, 116, 0.2); }
        #dropzone.has-image { border-style: solid; border-color: #d4a574; background: rgba(212, 165, 116, 0.08); }
        #dropzone p { color: #6b5742; }
        #dropzone .upload-icon { font-size: 48px; margin-bottom: 12px; }
        #dropzone p { font-size: 16px; margin-bottom: 4px; }
        #dropzone small { color: #c4b5fd; font-size: 12px; }
        #preview { max-height: 200px; border-radius: 12px; margin-top: 16px; display: none; }
        #file-input { display: none; }
        .btn { width: 100%; padding: 16px; margin-top: 20px; border: none; border-radius: 12px; font-size: 16px; font-weight: 600; cursor: pointer; background: #b8925f; color: #f5f1e8; transition: all 0.3s; }
        .btn:disabled { cursor: not-allowed; opacity: 0.5; }
        .btn.enabled { background: linear-gradient(135deg, #d4a574, #c9945e); color: #f5f1e8; font-weight: 700; box-shadow: 0 6px 25px rgba(212, 165, 116, 0.4); }
        .btn.enabled:hover { transform: translateY(-3px); box-shadow: 0 8px 35px rgba(212, 165, 116, 0.6); background: linear-gradient(135deg, #c9945e, #b8925f); }
        #result { margin-top: 20px; }
        .result-box { padding: 30px; border-radius: 16px; text-align: center; }
        .result-box.blocked { background: rgba(239,68,68,0.15); border: 2px solid rgba(239,68,68,0.4); }
        .result-box.success { background: rgba(34,197,94,0.15); border: 2px solid rgba(74,222,128,0.3); }
        .result-icon { font-size: 50px; margin-bottom: 12px; }
        .result-status { font-size: 20px; font-weight: 700; margin-bottom: 16px; }
        .result-box.blocked .result-status { color: #fca5a5; }
        .result-box.success .result-status { color: #5a8a5a; }
        .result-emotion { display: flex; align-items: center; justify-content: center; gap: 12px; }
        .emoji-large { font-size: 48px; }
        .emotion-name { font-size: 28px; font-weight: 600; text-transform: capitalize; color: #2c2416; }
        .confidence { color: #8b7355; font-size: 18px; }
        .warning-msg { margin-top: 20px; padding: 14px; background: rgba(234,179,8,0.15); border: 1px solid rgba(234,179,8,0.3); border-radius: 10px; color: #fef08a; font-size: 14px; }
        .error-box { padding: 20px; background: rgba(239,68,68,0.15); border: 1px solid rgba(248,113,113,0.4); border-radius: 12px; color: #fca5a5; font-size: 14px; }
        .probs-container { margin-top: 20px; background: rgba(255,255,255,0.05); border-radius: 12px; padding: 16px; }
        .probs-container h4 { font-size: 14px; margin-bottom: 14px; color: #e9d5ff; }
        .probs-list { display: flex; flex-direction: column; gap: 10px; }
        .prob-row { display: flex; align-items: center; gap: 10px; }
        .prob-emoji { font-size: 18px; width: 24px; }
        .prob-name { width: 80px; font-size: 13px; text-transform: capitalize; color: #e9d5ff; }
        .prob-name.text-blocked { color: #fca5a5; }
        .prob-bar-bg { flex: 1; height: 10px; background: rgba(255,255,255,0.1); border-radius: 5px; overflow: hidden; }
        .prob-bar { height: 100%; border-radius: 5px; transition: width 0.5s ease; }
        .bar-high { background: linear-gradient(90deg, #22c55e, #4ade80); }
        .bar-medium { background: linear-gradient(90deg, #eab308, #facc15); }
        .bar-low { background: #6b7280; }
        .bar-blocked { background: rgba(239,68,68,0.5); }
        .prob-value { width: 50px; text-align: right; font-size: 13px; color: #c4b5fd; }
        .prob-blocked { color: #f87171; font-size: 14px; }
        .footer { text-align: center; margin-top: 20px; padding: 24px; background: linear-gradient(135deg, rgba(255, 255, 255, 0.8), rgba(245, 241, 232, 0.9)); border-radius: 16px; border: 2px solid rgba(212, 165, 116, 0.3); box-shadow: 0 4px 20px rgba(139, 115, 85, 0.1); }
        .footer .credits { color: #6b5742; font-size: 16px; margin-bottom: 10px; font-weight: 700; }
        .footer .author { color: #8b7355; font-size: 14px; margin-bottom: 12px; font-weight: 500; }
        .footer .linkedin { display: inline-flex; align-items: center; gap: 8px; color: #f5f1e8; text-decoration: none; font-size: 14px; font-weight: 600; padding: 10px 20px; background: linear-gradient(135deg, #d4a574, #c9945e); border-radius: 8px; transition: all 0.3s; box-shadow: 0 4px 15px rgba(212, 165, 116, 0.3); }
        .footer .linkedin:hover { background: linear-gradient(135deg, #c9945e, #b8925f); transform: translateY(-3px); box-shadow: 0 6px 25px rgba(212, 165, 116, 0.5); }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-badge"><span>🎯</span><span>AI-Powered Moderation</span></div>
            <h1>AIM - Automated Image Moderator</h1>
            <p class="subtitle">Intelligent content filtering with emotion and safety detection</p>
        </div>

        <div class="main-grid">
            <div class="panel">
                <div class="tabs">
                    <button id="tab-emotions" class="tab active" onclick="showTab('emotions')">😊 Emotions <span id="emotion-count" class="count">5</span></button>
                    <button id="tab-objects" class="tab" onclick="showTab('objects')">⚔️ Objects <span id="object-count" class="count">10</span></button>
                </div>
                <div id="emotions-panel" class="filter-panel">
                    <!-- Emotions will be rendered here by JavaScript -->
                </div>
                <div id="objects-panel" class="filter-panel" style="display: none;">
                    <!-- Objects will be rendered here by JavaScript -->
                </div>
                <div class="warning-note"><p>⚠️ Blocked items will prevent upload and show a safety warning.</p></div>
            </div>

            <div class="panel upload-panel">
                <div id="dropzone" onclick="document.getElementById('file-input').click()">
                    <input type="file" id="file-input" accept="image/*" onchange="handleFile(event)">
                    <div id="upload-text">
                        <div class="upload-icon">📤</div>
                        <p>Drop an image or click to upload</p>
                        <small>Supports JPG, PNG, WebP</small>
                    </div>
                    <img id="preview" alt="Preview">
                </div>
                <button id="analyze-btn" class="btn" disabled onclick="analyzeImage()">👁️ Analyze Content</button>
                <div id="result"></div>
            </div>
        </div>

        <div class="footer">
            <p class="credits">⚡ Built with AI & Passion</p>
            <p class="author">Developed by Vidhu Anand Munshi</p>
            <a href="https://www.linkedin.com/in/vidhu-anand-munshi-a272b028a" target="_blank" class="linkedin">
                <span>💼</span>
                <span>Connect on LinkedIn</span>
            </a>
        </div>
    </div>

    <script>
        const API_URL = "/predict";

        const emotions = { 
            happy: { emoji: '😊', desc: 'Joyful, positive', allowed: true }, 
            neutral: { emoji: '😐', desc: 'Calm, balanced', allowed: true }, 
            surprise: { emoji: '😲', desc: 'Shocked, amazed', allowed: true }, 
            sad: { emoji: '😢', desc: 'Unhappy, sorrowful', allowed: false }, 
            angry: { emoji: '😠', desc: 'Frustrated, hostile', allowed: false }, 
            fear: { emoji: '😨', desc: 'Scared, anxious', allowed: false }, 
            disgust: { emoji: '🤢', desc: 'Revulsion', allowed: false }, 
            contempt: { emoji: '😏', desc: 'Disdain, scorn', allowed: false } 
        };

        const objects = { 
            weapon: { emoji: '⚔️', desc: 'General weapons', allowed: false, severity: 'high' }, 
            gun: { emoji: '🔫', desc: 'Firearms', allowed: false, severity: 'high' }, 
            knife: { emoji: '🔪', desc: 'Knives, blades', allowed: false, severity: 'high' }, 
            blood: { emoji: '🩸', desc: 'Blood, bleeding', allowed: false, severity: 'high' }, 
            violence: { emoji: '💥', desc: 'Violent scenes', allowed: false, severity: 'high' }, 
            drugs: { emoji: '💊', desc: 'Illegal substances', allowed: false, severity: 'medium' }, 
            alcohol: { emoji: '🍺', desc: 'Alcoholic drinks', allowed: false, severity: 'low' }, 
            cigarette: { emoji: '🚬', desc: 'Smoking, tobacco', allowed: false, severity: 'low' }, 
            nudity: { emoji: '🚫', desc: 'Adult content', allowed: false, severity: 'high' }, 
            gore: { emoji: '☠️', desc: 'Graphic imagery', allowed: false, severity: 'high' } 
        };

        let selectedFile = null;

        function renderEmotions() {
            const panel = document.getElementById('emotions-panel');
            if (!panel) {
                console.error('emotions-panel not found');
                return;
            }

            panel.innerHTML = '';

            Object.entries(emotions).forEach(([key, val]) => {
                const btn = document.createElement('button');
                btn.className = `filter-btn ${val.allowed ? 'allowed' : 'blocked'}`;
                btn.innerHTML = `
                    <span class="emoji">${val.emoji}</span>
                    <div class="info">
                        <span class="name">${key}</span>
                        <span class="desc">${val.desc}</span>
                    </div>
                    <span class="status">${val.allowed ? '✅' : '❌'}</span>
                `;
                btn.onclick = function() { 
                    emotions[key].allowed = !emotions[key].allowed; 
                    renderEmotions(); 
                    updateCounts(); 
                };
                panel.appendChild(btn);
            });

            console.log('Emotions rendered:', Object.keys(emotions).length);
        }

        function renderObjects() {
            const panel = document.getElementById('objects-panel');
            if (!panel) {
                console.error('objects-panel not found');
                return;
            }

            panel.innerHTML = '';

            Object.entries(objects).forEach(([key, val]) => {
                const btn = document.createElement('button');
                btn.className = `filter-btn ${val.allowed ? 'allowed' : 'blocked-' + val.severity}`;
                btn.innerHTML = `
                    <span class="emoji">${val.emoji}</span>
                    <div class="info">
                        <span class="name">${key}</span>
                        <span class="desc">${val.desc}</span>
                    </div>
                    ${val.severity === 'high' && !val.allowed ? '<span class="badge-high">HIGH</span>' : ''}
                    <span class="status">${val.allowed ? '✅' : '❌'}</span>
                `;
                btn.onclick = function() { 
                    objects[key].allowed = !objects[key].allowed; 
                    renderObjects(); 
                    updateCounts(); 
                };
                panel.appendChild(btn);
            });

            console.log('Objects rendered:', Object.keys(objects).length);
        }

        function updateCounts() {
            const emotionCount = document.getElementById('emotion-count');
            const objectCount = document.getElementById('object-count');
            if (emotionCount) emotionCount.textContent = Object.values(emotions).filter(e => !e.allowed).length;
            if (objectCount) objectCount.textContent = Object.values(objects).filter(o => !o.allowed).length;
        }

        function showTab(tab) {
            const emotionsPanel = document.getElementById('emotions-panel');
            const objectsPanel = document.getElementById('objects-panel');
            const tabEmotions = document.getElementById('tab-emotions');
            const tabObjects = document.getElementById('tab-objects');

            if (emotionsPanel) emotionsPanel.style.display = tab === 'emotions' ? 'block' : 'none';
            if (objectsPanel) objectsPanel.style.display = tab === 'objects' ? 'block' : 'none';
            if (tabEmotions) tabEmotions.className = `tab ${tab === 'emotions' ? 'active' : ''}`;
            if (tabObjects) tabObjects.className = `tab ${tab === 'objects' ? 'active' : ''}`;

            console.log('Switched to tab:', tab);
        }

        function handleFile(event) {
            const file = event.target.files[0];
            if (!file) return;
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                document.getElementById('preview').src = e.target.result;
                document.getElementById('preview').style.display = 'block';
                document.getElementById('upload-text').style.display = 'none';
                document.getElementById('dropzone').classList.add('has-image');
            };
            reader.readAsDataURL(file);
            document.getElementById('analyze-btn').disabled = false;
            document.getElementById('analyze-btn').className = 'btn enabled';
            document.getElementById('result').innerHTML = '';
        }

        async function analyzeImage() {
            if (!selectedFile) return;
            const btn = document.getElementById('analyze-btn');
            btn.textContent = '⏳ Analyzing...';
            btn.disabled = true;
            const formData = new FormData();
            formData.append('file', selectedFile);
            try {
                const res = await fetch(API_URL, { method: 'POST', body: formData });
                const data = await res.json();
                if (data.status === 'success') {
                    const prediction = data.prediction.toLowerCase();

                    // Check if prediction is an emotion or object
                    let isEmotionBlocked = false;
                    let isObjectBlocked = false;
                    let blockedObjects = [];

                    // Check if the main prediction is a blocked emotion
                    if (emotions[prediction]) {
                        isEmotionBlocked = !emotions[prediction].allowed;
                    }

                    // Check if the main prediction is a blocked object
                    if (objects[prediction]) {
                        isObjectBlocked = !objects[prediction].allowed;
                        if (isObjectBlocked) {
                            blockedObjects.push({
                                label: prediction,
                                confidence: data.confidence
                            });
                        }
                    }

                    // Check detected_objects array if it exists
                    if (data.detected_objects && data.detected_objects.length > 0) {
                        data.detected_objects.forEach(obj => {
                            const objName = obj.label.toLowerCase();
                            if (objects[objName] && !objects[objName].allowed) {
                                isObjectBlocked = true;
                                // Only add if not already added
                                if (!blockedObjects.find(b => b.label === objName)) {
                                    blockedObjects.push(obj);
                                }
                            }
                        });
                    }

                    // If either emotion or object is blocked, show blocked message
                    const isBlocked = isEmotionBlocked || isObjectBlocked;
                    console.log('Prediction:', prediction, 'IsBlocked:', isBlocked, 'BlockedObjects:', blockedObjects);
                    displayResult(prediction, data, isBlocked, blockedObjects);
                } else {
                    displayError(data.reason || 'Failed');
                }
            } catch (err) {
                displayError('Cannot connect to server: ' + err.message);
            }
            btn.textContent = '👁️ Analyze Content';
            btn.disabled = false;
            btn.className = 'btn enabled';
        }

        function displayResult(prediction, data, isBlocked, blockedObjects = []) {
            // Get emoji from emotions or objects
            const emoji = emotions[prediction]?.emoji || objects[prediction]?.emoji || '🎭';
            const confidence = (data.confidence * 100).toFixed(1);

            let blockReasons = [];

            // If prediction is a blocked emotion
            if (emotions[prediction] && !emotions[prediction].allowed) {
                blockReasons.push(`Emotion: ${prediction}`);
            }

            // If prediction is a blocked object
            if (objects[prediction] && !objects[prediction].allowed) {
                blockReasons.push(`${prediction} (${confidence}%)`);
            }

            // Add any additional blocked objects
            if (blockedObjects.length > 0) {
                blockedObjects.forEach(obj => {
                    if (!blockReasons.includes(`${obj.label} (${(obj.confidence * 100).toFixed(1)}%)`)) {
                        blockReasons.push(`${obj.label} (${(obj.confidence * 100).toFixed(1)}%)`);
                    }
                });
            }

            let html = `<div class="result-box ${isBlocked ? 'blocked' : 'success'}">
                <div class="result-icon">${isBlocked ? '❌' : '✅'}</div>
                <div class="result-status">${isBlocked ? 'UPLOAD BLOCKED' : 'Safe to Upload'}</div>
                <div class="result-emotion">
                    <span class="emoji-large">${emoji}</span>
                    <span class="emotion-name">${prediction}</span>
                    <span class="confidence">${confidence}%</span>
                </div>
                ${isBlocked ? `<div class="warning-msg">⚠️ <strong>Blocked Content Detected:</strong><br>${blockReasons.join(', ')}<br><br>This content violates your filter settings. Enable it in the filters panel to allow upload.</div>` : ''}
            </div>`;

            html += '<div class="probs-container"><h4>All Predictions</h4><div class="probs-list">';
            const sorted = Object.entries(data.all_probs).sort((a, b) => b[1] - a[1]);
            sorted.forEach(([label, prob]) => {
                // Check if this label is allowed in either emotions or objects
                const isEmotionAllowed = emotions[label] ? emotions[label].allowed : true;
                const isObjectAllowed = objects[label] ? objects[label].allowed : true;
                const isAllowed = isEmotionAllowed && isObjectAllowed;

                const pct = (prob * 100).toFixed(1);
                const barClass = !isAllowed ? 'bar-blocked' : (prob > 0.7 ? 'bar-high' : prob > 0.4 ? 'bar-medium' : 'bar-low');
                const displayEmoji = emotions[label]?.emoji || objects[label]?.emoji || '🎭';

                html += `<div class="prob-row">
                    <span class="prob-emoji">${displayEmoji}</span>
                    <span class="prob-name ${!isAllowed ? 'text-blocked' : ''}">${label}</span>
                    <div class="prob-bar-bg">
                        <div class="prob-bar ${barClass}" style="width: ${pct}%"></div>
                    </div>
                    <span class="prob-value">${pct}%</span>
                    ${!isAllowed ? '<span class="prob-blocked">❌</span>' : ''}
                </div>`;
            });
            html += '</div></div>';
            document.getElementById('result').innerHTML = html;
        }

        function displayError(message) {
            document.getElementById('result').innerHTML = `<div class="error-box">❌ ${message}</div>`;
        }

        function setupDragAndDrop() {
            const dropzone = document.getElementById('dropzone');
            if (!dropzone) return;
            dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('drag-over'); });
            dropzone.addEventListener('dragleave', () => { dropzone.classList.remove('drag-over'); });
            dropzone.addEventListener('drop', (e) => { e.preventDefault(); dropzone.classList.remove('drag-over'); const files = e.dataTransfer.files; if (files.length > 0) { document.getElementById('file-input').files = files; handleFile({ target: { files: files } }); } });
        }

        // Initialize when page loads
        window.addEventListener('DOMContentLoaded', function() {
            console.log('Page loaded, initializing...');
            renderEmotions();
            renderObjects();
            updateCounts();
            setupDragAndDrop();
            console.log('✅ Frontend initialized');
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


# ======== RUN SERVER ========
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting Emotion Detection Server")
    print("🌐 Open your browser and go to: http://localhost:8000")
    print("📡 API Endpoint: POST /predict")
    print("=" * 60)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)