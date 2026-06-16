"""
Weather Classification — ResNet18 Transfer Learning
====================================================
Dataset : https://www.kaggle.com/datasets/jehanbhathena/weather-dataset
          12 classes: dew, fogsmog, frost, glaze, hail, lightning,
                      rain, rainbow, rime, sandstorm, snow, sunrise

HOW TO RUN:
  1. Download the Kaggle dataset and unzip it.
  2. Place the 'Weather Dataset' folder next to this script.
  3. Run:  python train.py
  4. Best model saved to:  model/weather_transfer_best.pth
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# ─────────────────────────────────────────────────────────────────
# PATHS
# Weather Dataset/ must sit next to this script:
#   Weather Dataset/
#       dew/ fogsmog/ frost/ glaze/ hail/ lightning/
#       rain/ rainbow/ rime/ sandstorm/ snow/
# ─────────────────────────────────────────────────────────────────

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, 'Weather Dataset')

print(f"Dataset path : {DATA_DIR}")

# Verify folder exists before doing anything
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(
        f"\n✗ Dataset folder not found: {DATA_DIR}"
        "\n  → Download the Kaggle dataset and unzip it next to train.py"
    )

# ─────────────────────────────────────────────────────────────────
# TRANSFORMS
# ResNet18 was pretrained on ImageNet — use ImageNet mean/std exactly.
# ─────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# ─────────────────────────────────────────────────────────────────
# DATASET — 70% train, 15% validation, 15% test
# ─────────────────────────────────────────────────────────────────

full_dataset_aug   = torchvision.datasets.ImageFolder(root=DATA_DIR, transform=transform_train)
full_dataset_clean = torchvision.datasets.ImageFolder(root=DATA_DIR, transform=transform_test)

total      = len(full_dataset_aug)
train_size = int(0.70 * total)
val_size   = int(0.15 * total)
test_size  = total - train_size - val_size

# Use generator for reproducibility
generator = torch.Generator().manual_seed(42)
indices = torch.randperm(total, generator=generator).tolist()

train_indices = indices[:train_size]
val_indices   = indices[train_size:train_size + val_size]
test_indices  = indices[train_size + val_size:]

train_dataset = Subset(full_dataset_aug,   train_indices)
val_dataset   = Subset(full_dataset_clean, val_indices)
test_dataset  = Subset(full_dataset_clean, test_indices)

class_names = full_dataset_aug.classes
num_classes = len(class_names)

print(f"\nDataset sizes:")
print(f"  Train      : {len(train_dataset)} images")
print(f"  Validation : {len(val_dataset)} images")
print(f"  Test       : {len(test_dataset)} images")
print(f"  Classes ({num_classes}): {class_names}")

# ─────────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────────

use_gpu = torch.cuda.is_available()
pin_mem = True if use_gpu else False
num_workers = 0 if os.name == 'nt' else 2

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=num_workers, pin_memory=pin_mem)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

device = torch.device('cuda' if use_gpu else 'cpu')
print(f"\nDevice: {device}")

# ─────────────────────────────────────────────────────────────────
# RESNET18 WITH TRANSFER LEARNING
#
# Freezing strategy for weather images (similar to ImageNet domain):
#   conv1, layer1, layer2 → FREEZE (universal low-level features)
#   layer3, layer4        → UNFREEZE (fine-tune for weather patterns)
#   fc                    → REPLACE (new output layer for our classes)
# ─────────────────────────────────────────────────────────────────

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze early layers
for param in model.conv1.parameters():  param.requires_grad = False
for param in model.layer1.parameters(): param.requires_grad = False
for param in model.layer2.parameters(): param.requires_grad = False

# Unfreeze deeper layers
for param in model.layer3.parameters(): param.requires_grad = True
for param in model.layer4.parameters(): param.requires_grad = True

# Replace output layer
in_features = model.fc.in_features   # 512 for ResNet18
model.fc = nn.Linear(in_features, num_classes)
model = model.to(device)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params    = total_params - trainable_params

print(f"\nModel: ResNet18 Transfer Learning")
print(f"  Total params     : {total_params:,}")
print(f"  Trainable params : {trainable_params:,}  ← update during training")
print(f"  Frozen params    : {frozen_params:,}  ← locked pretrained knowledge")

# ─────────────────────────────────────────────────────────────────
# LOSS, OPTIMIZER, SCHEDULER
# ─────────────────────────────────────────────────────────────────

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4   # small LR — careful fine-tuning
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.1
)

# ─────────────────────────────────────────────────────────────────
# TRAIN / EVALUATE FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    correct = 0; total_loss = 0
    total = len(loader)
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (outputs.argmax(1) == labels).sum().item()
        done = int(30 * (batch_idx + 1) / total)
        print(f"\r  Batch {batch_idx+1}/{total} [{'█'*done}{'░'*(30-done)}]", end='')
    print()
    return total_loss / len(loader), correct / len(loader.dataset) * 100

def evaluate(model, loader, criterion, device):
    model.eval()
    correct = 0; total_loss = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct    += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset) * 100

# ─────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────

EPOCHS = 10
best_val_acc = 0.0

MODEL_DIR  = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'weather_transfer_best.pth')
os.makedirs(MODEL_DIR, exist_ok=True)

train_accs = []; val_accs = []; train_losses = []; val_losses = []

print("\nStarting Training...")
print("─" * 65)

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.2e}")
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)

    train_accs.append(train_acc);   val_accs.append(val_acc)
    train_losses.append(train_loss); val_losses.append(val_loss)

    scheduler.step(val_loss)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        saved = "  ✓ best model saved"
    else:
        saved = ""

    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%{saved}")

print("\n" + "─" * 65)
print(f"Best Validation Accuracy : {best_val_acc:.2f}%")
print(f"Model saved to           : {MODEL_PATH}")

# ─────────────────────────────────────────────────────────────────
# FINAL TEST EVALUATION
# ─────────────────────────────────────────────────────────────────

print("\nLoading best model for final test evaluation...")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Final Test Accuracy: {test_acc:.2f}%")

# ─────────────────────────────────────────────────────────────────
# PLOTS — Accuracy & Loss curves
# ─────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(train_accs,   marker='o', label='Train Accuracy')
ax1.plot(val_accs,     marker='o', label='Val Accuracy')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Train vs Validation Accuracy'); ax1.legend()

ax2.plot(train_losses, marker='o', label='Train Loss')
ax2.plot(val_losses,   marker='o', label='Val Loss')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
ax2.set_title('Train vs Validation Loss'); ax2.legend()

plt.suptitle('Weather Classification — ResNet18 Transfer Learning')
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────

model.eval()
all_preds = []; all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

cm   = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(12, 10))
disp.plot(cmap='Blues', ax=ax)
plt.title('ResNet18 Transfer Learning — Confusion Matrix (Test Set)')
plt.tight_layout()
plt.show()
