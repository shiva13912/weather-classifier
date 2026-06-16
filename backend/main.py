import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io

# ─────────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────────

app = FastAPI(title="Weather Classifier API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In production, replace * with your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────
# CLASS NAMES
# These MUST match the folder names in your dataset, sorted A→Z.
# The Kaggle dataset (jehanbhathena/weather-dataset) has 12 classes:
# ─────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    'dew', 'fogsmog', 'frost', 'glaze', 'hail',
    'lightning', 'rain', 'rainbow', 'rime', 'sandstorm',
    'snow'
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "weather_transfer_best.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """Load ResNet18 with same architecture as training script."""
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features                        # 512
    model.fc = nn.Linear(in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

# Load model once at startup
try:
    model = load_model()
    print(f"✓ Model loaded from {MODEL_PATH}")
    print(f"✓ Running on: {device}")
    print(f"✓ Classes: {CLASS_NAMES}")
except FileNotFoundError:
    print(f"✗ Model file not found at {MODEL_PATH}")
    print("  → Run train.py first, then place weather_transfer_best.pth in the /model folder")
    model = None

# ─────────────────────────────────────────────────────────────────
# TRANSFORM — same as test transform in training
# ─────────────────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# ─────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "Weather Classifier API is running!", "status": "ok"}

@app.get("/health")
def health():
    return {
        "model_loaded": model is not None,
        "device": str(device),
        "classes": CLASS_NAMES
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload an image → get predicted weather class + confidence scores.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (jpg, png, etc.)")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0]

    pred_idx   = probs.argmax().item()
    pred_class = CLASS_NAMES[pred_idx]
    confidence = round(probs[pred_idx].item() * 100, 2)

    all_scores = {
        CLASS_NAMES[i]: round(probs[i].item() * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    return JSONResponse({
        "prediction": pred_class,
        "confidence": confidence,
        "all_scores": all_scores
    })
