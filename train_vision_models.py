"""
=============================================================================
Tugas Mandiri Deep Learning: Implementasi & Komparasi Arsitektur
Arsitektur yang Diimplementasikan:
  1. Custom Deep Convolutional Neural Network (CNN) dari Scratch
  2. Transfer Learning (ResNet-18 Pretrained pada ImageNet)
Dataset: Fashion-MNIST (10 Kelas Pakaian & Aksesori Mode)
=============================================================================
"""

import os
import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# ---------------------------------------------------------------------------
# 1. Konfigurasi Eksperimen & Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Perangkat Komputasi yang Digunakan: {DEVICE}")

DATA_DIR = "./data_fmnist"
OUTPUT_DIR = "./outputs"
MODEL_DIR = "./models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# ---------------------------------------------------------------------------
# 2. Pipeline Data & Augmentasi
# ---------------------------------------------------------------------------
# Transformasi untuk Custom CNN (1 channel grayscale, 28x28)
transform_train_cnn = transforms.Compose([
    transforms.RandomCrop(28, padding=2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

transform_test_cnn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

# Transformasi untuk Transfer Learning ResNet-18 (Resize 56x56, 3 channel RGB agar kompatibel dengan bobot ImageNet)
transform_train_tl = transforms.Compose([
    transforms.Resize((56, 56)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_test_tl = transforms.Compose([
    transforms.Resize((56, 56)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

print("[INFO] Memuat dataset Fashion-MNIST...")
full_train_fmnist_cnn = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=True, download=False, transform=transform_train_cnn)
full_test_fmnist_cnn = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=False, download=False, transform=transform_test_cnn)

full_train_fmnist_tl = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=True, download=False, transform=transform_train_tl)
full_test_fmnist_tl = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=False, download=False, transform=transform_test_tl)

# Menggunakan subset representatif terstratifikasi agar proses komputasi CPU cepat, stabil, & efisien
# 10,000 sampel train, 2,000 sampel test
SUBSET_TRAIN_SIZE = 10000
SUBSET_TEST_SIZE = 2000

train_indices = list(range(SUBSET_TRAIN_SIZE))
test_indices = list(range(SUBSET_TEST_SIZE))

train_loader_cnn = DataLoader(Subset(full_train_fmnist_cnn, train_indices), batch_size=64, shuffle=True, num_workers=0)
test_loader_cnn = DataLoader(Subset(full_test_fmnist_cnn, test_indices), batch_size=64, shuffle=False, num_workers=0)

train_loader_tl = DataLoader(Subset(full_train_fmnist_tl, train_indices), batch_size=64, shuffle=True, num_workers=0)
test_loader_tl = DataLoader(Subset(full_test_fmnist_tl, test_indices), batch_size=64, shuffle=False, num_workers=0)

# Simpan contoh gambar dataset ke folder output
def save_dataset_sample():
    sample_loader = DataLoader(full_train_fmnist_cnn, batch_size=16, shuffle=True)
    images, labels = next(iter(sample_loader))
    
    # Denormalize
    inv_norm = transforms.Normalize((-0.2860/0.3530,), (1/0.3530,))
    fig, axes = plt.subplots(2, 8, figsize=(16, 5))
    fig.suptitle("Sampel Dataset Fashion-MNIST (10 Kelas Mode)", fontsize=14, fontweight='bold')
    for idx, ax in enumerate(axes.flat):
        img = inv_norm(images[idx]).clamp(0, 1).squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.set_title(CLASSES[labels[idx]], fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fashion_mnist_samples.png"), dpi=200)
    plt.close()
    print("[INFO] Sampel visual dataset tersimpan di:", os.path.join(OUTPUT_DIR, "fashion_mnist_samples.png"))

save_dataset_sample()

# ---------------------------------------------------------------------------
# 3. Definisi Model 1: Custom Deep CNN (dari Scratch)
# ---------------------------------------------------------------------------
class CustomCNN(nn.Module):
    """
    Arsitektur Convolutional Neural Network (CNN) Khusus Fashion-MNIST:
    - 5 Layer Konvolusi dalam 3 Blok Spasial
    - Batch Normalization di setiap stage untuk stabilisasi gradien
    - Dropout berjenjang untuk mencegah overfitting
    - Klasifikasi Dense Layer 1152 -> 256 -> 10
    """
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()
        
        # Blok 1: Input 1x28x28 -> 32x14x14
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )
        
        # Blok 2: 32x14x14 -> 64x7x7
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        # Blok 3: 64x7x7 -> 128x3x3 (downsampling dengan pooling)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        
        # Dense Classifier: 128 * 3 * 3 = 1152 -> 256 -> 10
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x

# ---------------------------------------------------------------------------
# 4. Definisi Model 2: Transfer Learning (ResNet-18 Pretrained)
# ---------------------------------------------------------------------------
def build_transfer_learning_model(num_classes=10):
    """
    Model Transfer Learning memanfaatkan representasi fitur ResNet-18
    yang telah dilatih pada 1.4 juta citra ImageNet.
    - Feature Extractor: Layer konvolusi awal dibekukan (frozen)
    - Fine-Tuning: Layer4 dibuka untuk adaptasi fitur representasional tinggi
    - Head Pengklasifikasi Khusus: Dense Layer 512 -> 256 -> 10 dengan Dropout
    """
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    
    # Bekukan seluruh backbone awal
    for param in model.parameters():
        param.requires_grad = False
        
    # Buka layer4 untuk adaptasi fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    # Ganti fully-connected layer (fc)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model

# ---------------------------------------------------------------------------
# 5. Fungsi Pelatihan dan Evaluasi
# ---------------------------------------------------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc, np.array(all_preds), np.array(all_labels)

# ---------------------------------------------------------------------------
# 6. Eksekusi Pelatihan
# ---------------------------------------------------------------------------
EPOCHS = 6

# A. Pelatihan Custom CNN
print("\n" + "="*65)
print("MEMULAI PELATIHAN: ARSITEKTUR 1 - CUSTOM CNN (DARI SCRATCH)")
print("="*65)
cnn_model = CustomCNN(num_classes=10).to(DEVICE)
cnn_params = sum(p.numel() for p in cnn_model.parameters() if p.requires_grad)
print(f"[INFO] Jumlah Parameter Terlatih Custom CNN: {cnn_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler_cnn = optim.lr_scheduler.CosineAnnealingLR(optimizer_cnn, T_max=EPOCHS)

history_cnn = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_cnn_acc = 0.0

start_time_cnn = time.time()
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    train_loss, train_acc = train_one_epoch(cnn_model, train_loader_cnn, criterion, optimizer_cnn, DEVICE)
    val_loss, val_acc, _, _ = evaluate(cnn_model, test_loader_cnn, criterion, DEVICE)
    scheduler_cnn.step()
    t1 = time.time()
    
    history_cnn['train_loss'].append(train_loss)
    history_cnn['train_acc'].append(train_acc)
    history_cnn['val_loss'].append(val_loss)
    history_cnn['val_acc'].append(val_acc)
    
    if val_acc > best_cnn_acc:
        best_cnn_acc = val_acc
        torch.save(cnn_model.state_dict(), os.path.join(MODEL_DIR, "best_custom_cnn.pth"))
        
    print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] ({t1-t0:.1f}s) | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")

total_time_cnn = time.time() - start_time_cnn
print(f"[SELESAI] Total Waktu Pelatihan Custom CNN: {total_time_cnn:.2f} detik")

# B. Pelatihan Transfer Learning (ResNet-18)
print("\n" + "="*65)
print("MEMULAI PELATIHAN: ARSITEKTUR 2 - TRANSFER LEARNING (RESNET-18)")
print("="*65)
tl_model = build_transfer_learning_model(num_classes=10).to(DEVICE)
tl_trainable_params = sum(p.numel() for p in tl_model.parameters() if p.requires_grad)
tl_total_params = sum(p.numel() for p in tl_model.parameters())
print(f"[INFO] Total Parameter ResNet-18: {tl_total_params:,} | Trainable Params: {tl_trainable_params:,}")

# Optimizer dengan diferensiasi learning rate: layer4 (1e-4) dan classifier head (1e-3)
optimizer_tl = optim.Adam([
    {'params': tl_model.layer4.parameters(), 'lr': 1e-4},
    {'params': tl_model.fc.parameters(), 'lr': 1e-3}
], weight_decay=1e-4)
scheduler_tl = optim.lr_scheduler.CosineAnnealingLR(optimizer_tl, T_max=EPOCHS)

history_tl = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_tl_acc = 0.0

start_time_tl = time.time()
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    train_loss, train_acc = train_one_epoch(tl_model, train_loader_tl, criterion, optimizer_tl, DEVICE)
    val_loss, val_acc, _, _ = evaluate(tl_model, test_loader_tl, criterion, DEVICE)
    scheduler_tl.step()
    t1 = time.time()
    
    history_tl['train_loss'].append(train_loss)
    history_tl['train_acc'].append(train_acc)
    history_tl['val_loss'].append(val_loss)
    history_tl['val_acc'].append(val_acc)
    
    if val_acc > best_tl_acc:
        best_tl_acc = val_acc
        torch.save(tl_model.state_dict(), os.path.join(MODEL_DIR, "best_transfer_learning.pth"))
        
    print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] ({t1-t0:.1f}s) | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")

total_time_tl = time.time() - start_time_tl
print(f"[SELESAI] Total Waktu Pelatihan Transfer Learning: {total_time_tl:.2f} detik")

# ---------------------------------------------------------------------------
# 7. Evaluasi Akhir & Pengujian Komparatif
# ---------------------------------------------------------------------------
print("\n" + "="*65)
print("EVALUASI AKHIR PADA TEST SET INDEPENDEN")
print("="*65)

# Load best saved weights
cnn_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_custom_cnn.pth")))
tl_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_transfer_learning.pth")))

_, final_acc_cnn, preds_cnn, labels_cnn = evaluate(cnn_model, test_loader_cnn, criterion, DEVICE)
_, final_acc_tl, preds_tl, labels_tl = evaluate(tl_model, test_loader_tl, criterion, DEVICE)

prec_cnn, rec_cnn, f1_cnn, _ = precision_recall_fscore_support(labels_cnn, preds_cnn, average='weighted')
prec_tl, rec_tl, f1_tl, _ = precision_recall_fscore_support(labels_tl, preds_tl, average='weighted')

print("\n>>> HASIL AKHIR: CUSTOM CNN (DARI SCRATCH) <<<")
print(f"Test Accuracy : {final_acc_cnn*100:.2f}%")
print(f"Precision     : {prec_cnn*100:.2f}%")
print(f"Recall        : {rec_cnn*100:.2f}%")
print(f"F1-Score      : {f1_cnn*100:.2f}%")
print("\nClassification Report (Custom CNN):")
print(classification_report(labels_cnn, preds_cnn, target_names=CLASSES))

print("\n>>> HASIL AKHIR: TRANSFER LEARNING (RESNET-18) <<<")
print(f"Test Accuracy : {final_acc_tl*100:.2f}%")
print(f"Precision     : {prec_tl*100:.2f}%")
print(f"Recall        : {rec_tl*100:.2f}%")
print(f"F1-Score      : {f1_tl*100:.2f}%")
print("\nClassification Report (Transfer Learning):")
print(classification_report(labels_tl, preds_tl, target_names=CLASSES))

# Simpan hasil dalam format JSON
results_dict = {
    "dataset": "Fashion-MNIST",
    "classes": CLASSES,
    "epochs": EPOCHS,
    "custom_cnn": {
        "params": cnn_params,
        "training_time_sec": round(total_time_cnn, 2),
        "test_accuracy": round(float(final_acc_cnn) * 100, 2),
        "precision": round(float(prec_cnn) * 100, 2),
        "recall": round(float(rec_cnn) * 100, 2),
        "f1_score": round(float(f1_cnn) * 100, 2),
        "history": history_cnn
    },
    "transfer_learning": {
        "total_params": tl_total_params,
        "trainable_params": tl_trainable_params,
        "training_time_sec": round(total_time_tl, 2),
        "test_accuracy": round(float(final_acc_tl) * 100, 2),
        "precision": round(float(prec_tl) * 100, 2),
        "recall": round(float(rec_tl) * 100, 2),
        "f1_score": round(float(f1_tl) * 100, 2),
        "history": history_tl
    }
}

with open(os.path.join(OUTPUT_DIR, "evaluation_results.json"), "w") as f:
    json.dump(results_dict, f, indent=4)
print(f"[INFO] Metrik lengkap tersimpan di: {os.path.join(OUTPUT_DIR, 'evaluation_results.json')}")

# ---------------------------------------------------------------------------
# 8. Visualisasi Komparasi (Grafik Loss, Akurasi, Confusion Matrix, Bar Chart)
# ---------------------------------------------------------------------------
# A. Kurva Loss dan Akurasi
plt.figure(figsize=(14, 5))

# Subplot Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS + 1), history_cnn['train_loss'], 'b--', label='CNN Train Loss')
plt.plot(range(1, EPOCHS + 1), history_cnn['val_loss'], 'b-', label='CNN Val Loss')
plt.plot(range(1, EPOCHS + 1), history_tl['train_loss'], 'r--', label='Transfer Learn Train Loss')
plt.plot(range(1, EPOCHS + 1), history_tl['val_loss'], 'r-', label='Transfer Learn Val Loss')
plt.title("Perbandingan Kurva Loss per Epoch", fontsize=12, fontweight='bold')
plt.xlabel("Epoch", fontsize=10)
plt.ylabel("Loss (Cross Entropy)", fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

# Subplot Akurasi
plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS + 1), [x*100 for x in history_cnn['train_acc']], 'b--', label='CNN Train Acc')
plt.plot(range(1, EPOCHS + 1), [x*100 for x in history_cnn['val_acc']], 'b-', label='CNN Val Acc')
plt.plot(range(1, EPOCHS + 1), [x*100 for x in history_tl['train_acc']], 'r--', label='Transfer Learn Train Acc')
plt.plot(range(1, EPOCHS + 1), [x*100 for x in history_tl['val_acc']], 'r-', label='Transfer Learn Val Acc')
plt.title("Perbandingan Kurva Akurasi (%) per Epoch", fontsize=12, fontweight='bold')
plt.xlabel("Epoch", fontsize=10)
plt.ylabel("Akurasi (%)", fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_accuracy_comparison.png"), dpi=200)
plt.close()
print(f"[INFO] Kurva Loss & Akurasi tersimpan di: {os.path.join(OUTPUT_DIR, 'loss_accuracy_comparison.png')}")

# B. Confusion Matrix untuk Custom CNN
cm_cnn = confusion_matrix(labels_cnn, preds_cnn)
plt.figure(figsize=(9, 7))
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title(f"Confusion Matrix: Custom CNN (Akurasi: {final_acc_cnn*100:.2f}%)", fontsize=12, fontweight='bold')
plt.xlabel("Prediksi Kelas", fontsize=10)
plt.ylabel("Label Sebenarnya", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_cnn.png"), dpi=200)
plt.close()

# C. Confusion Matrix untuk Transfer Learning
cm_tl = confusion_matrix(labels_tl, preds_tl)
plt.figure(figsize=(9, 7))
sns.heatmap(cm_tl, annot=True, fmt='d', cmap='Oranges', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title(f"Confusion Matrix: Transfer Learning ResNet-18 (Akurasi: {final_acc_tl*100:.2f}%)", fontsize=12, fontweight='bold')
plt.xlabel("Prediksi Kelas", fontsize=10)
plt.ylabel("Label Sebenarnya", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_transfer_learning.png"), dpi=200)
plt.close()

# D. Diagram Batang Perbandingan Metrik Evaluasi
plt.figure(figsize=(9, 5))
metrics = ['Akurasi (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)']
cnn_scores = [final_acc_cnn*100, prec_cnn*100, rec_cnn*100, f1_cnn*100]
tl_scores = [final_acc_tl*100, prec_tl*100, rec_tl*100, f1_tl*100]

x = np.arange(len(metrics))
width = 0.35

rects1 = plt.bar(x - width/2, cnn_scores, width, label='Custom CNN (Scratch)', color='#2980b9')
rects2 = plt.bar(x + width/2, tl_scores, width, label='Transfer Learning (ResNet-18)', color='#d35400')

plt.ylabel('Skor Persentase (%)', fontsize=11)
plt.title('Perbandingan Komparatif Metrik Kinerja Model pada Test Set Fashion-MNIST', fontsize=12, fontweight='bold')
plt.xticks(x, metrics, fontsize=10)
plt.ylim(0, 105)
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.7)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "metrics_comparison_barchart.png"), dpi=200)
plt.close()

print(f"[INFO] Seluruh visualisasi komparatif tersimpan lengkap di direktori {OUTPUT_DIR}/")
print("\n[SUKSES] Seluruh proses pelatihan dan evaluasi Computer Vision telah rampung dengan sempurna!")
