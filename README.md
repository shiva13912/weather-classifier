# 🌦️ Weather Classifier — ResNet18 Transfer Learning

A beginner-friendly full-stack ML project.

| Layer | Tech |
|---|---|
| Model | ResNet18 fine-tuned on 11 weather classes (PyTorch) |
| Backend | FastAPI REST API |
| Frontend | Plain HTML/CSS/JS — drag-and-drop image upload |

---

## 📁 Project Structure

```
weather-classifier/
│
├── train.py                        ← Run this FIRST to train
├── README.md
├── .gitignore
│
├── model/
│   └── weather_transfer_best.pth   ← Created after training
│
├── backend/
│   ├── main.py                     ← FastAPI server
│   └── requirements.txt
│
├── frontend/
│   └── index.html                  ← Open in browser
│
└── Weather Dataset/                ← Placed next to train.py
    ├── dew/  fogsmog/  frost/  glaze/  hail/  lightning/
    └── rain/ rainbow/  rime/   sandstorm/ snow/
```

---

## ✅ Step-by-Step Setup

### STEP 1 — Verify the Dataset
The dataset is located in the `Weather Dataset/` folder next to `train.py`. It contains the 11 class folders directly. No manual train/test split folders are required.

---

### STEP 2 — Activate the Virtual Environment

Open a terminal in the project folder and run:

```powershell
# Activate the virtual environment (Windows)
venv\Scripts\activate

# Install all dependencies (already pre-installed in venv)
pip install matplotlib scikit-learn -r backend/requirements.txt
```

---

### STEP 3 — Train the Model

```bash
python train.py
```

This will:
- Load all 11 weather classes from `Weather Dataset`
- Automatically split the dataset in code: 70% Train, 15% Validation, 15% Test
- Fine-tune ResNet18 for 10 epochs
- Save the best model to `model/weather_transfer_best.pth`
- Show accuracy/loss plots + confusion matrix

⏱️ Time: ~5 min with GPU · ~30–45 min on CPU

---

### STEP 4 — Start the Backend

```bash
cd backend
uvicorn main:app --reload
```

Expected output:
```
✓ Model loaded from ../model/weather_transfer_best.pth
✓ Running on: cpu
✓ Classes: ['dew', 'fogsmog', 'frost', ...]
INFO: Uvicorn running on http://127.0.0.1:8000
```

Test it: open http://127.0.0.1:8000 in your browser.

---

### STEP 5 — Open the Frontend

Double-click `frontend/index.html` — it opens in your browser.

- Drag any weather photo onto the page
- Click **Classify Weather**
- See prediction + all 11 class confidence bars

---

## 🐙 Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit — weather classifier"

# Create a repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/weather-classifier.git
git branch -M main
git push -u origin main
```

> The `Weather Dataset/` folder and `model/*.pth` weights files are ignored by `.gitignore` so they won't be committed to Git.

---

## 🌐 Free Deployment

### Backend → Render.com
1. New → Web Service → connect GitHub repo
2. Root Directory: `backend`
3. Build: `pip install -r requirements.txt`
4. Start: `uvicorn main:app --host 0.0.0.0 --port 10000`
5. Update `API_URL` in `frontend/index.html` to your Render URL

### Frontend → GitHub Pages
1. Repo Settings → Pages → Source: `main` branch, `/frontend` folder
2. URL: `https://YOUR_USERNAME.github.io/weather-classifier`

---

## 🔧 Common Issues

| Problem | Fix |
|---|---|
| `FileNotFoundError: Weather Dataset` | Ensure `Weather Dataset` folder is placed next to `train.py` |
| `Model not loaded` API error | Run `train.py` first to generate the `.pth` file |
| CORS error in browser | Make sure FastAPI is running on port 8000 |
| `No module named torch` | Activate venv and run `pip install torch torchvision` |
| Low accuracy | Try 15–20 epochs or unfreeze `layer2` in `train.py` |

---

## 🎯 11 Weather Classes

`dew` · `fogsmog` · `frost` · `glaze` · `hail` · `lightning` · `rain` · `rainbow` · `rime` · `sandstorm` · `snow`
