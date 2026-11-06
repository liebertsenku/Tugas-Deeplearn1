"""
=============================================================================
Tugas Mandiri Deep Learning: Implementasi & Komparasi Arsitektur
Arsitektur yang Diimplementasikan:
  1. Simple RNN (Recurrent Neural Network / Elman RNN)
  2. GRU (Gated Recurrent Unit)
Dataset: Sentiment Analysis (Klasifikasi Sentimen Teks Ulasan / NLP)
=============================================================================
"""

import os
import time
import json
import random
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# ---------------------------------------------------------------------------
# 1. Konfigurasi Reproducibility & Device
# ---------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO-RNN/GRU] Perangkat Komputasi yang Digunakan: {DEVICE}")

OUTPUT_DIR = "./outputs"
MODEL_DIR = "./models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Pembuatan Dataset Sentimen Teks Realistis (Bahasa Indonesia & Multidomain)
# ---------------------------------------------------------------------------
pos_templates = [
    "produk ini sangat bagus dan berfungsi dengan sempurna kualitas terbaik",
    "pelayanan sangat cepat pengiriman aman dan respon penjual sangat ramah memuaskan",
    "luar biasa mantap sekali barang sesuai pesanan sangat direkomendasikan untuk dibeli",
    "desain elegan bahan berkualitas tinggi harga terjangkau sangat puas dengan pembelian ini",
    "performa sangat kencang baterai tahan lama dan layar sangat jernih luar biasa",
    "sangat suka sekali pengemasan rapi kurir ramah bintang lima untuk toko ini",
    "hasilnya melebihi ekspektasi sangat membantu pekerjaan sehari-hari mantap",
    "original asli bergaransi resmi tidak ada cacat sedikitpun sangat senang belanja di sini",
    "aplikasi sangat responsif fitur lengkap antarmuka intuitif dan mudah dipahami",
    "film ini sangat menyentuh alur cerita mendalam akting memukau dan pesan moral kuat"
]

neg_templates = [
    "sangat mengecewakan barang rusak saat tiba dan tidak bisa digunakan sama sekali",
    "kualitas sangat buruk bahan tipis mudah patah jangan beli produk ini",
    "pengiriman sangat lambat respon penjual sangat lambat dan tidak bertanggung jawab",
    "tidak sesuai deskripsi barang cacat warna pudar sangat rugi beli barang ini",
    "baterai sangat boros cepat panas sistem sering mengalami crash dan lambat",
    "kecewa sekali packing hanya plastik tipis barang penyok komplain diabaikan",
    "kualitas abal-abal harga mahal tapi tidak awet sangat menyesal membeli ini",
    "antarmuka rumit banyak bug sering force close dan sangat menyebalkan",
    "aktingnya kaku alur cerita membosankan tanpa arah film yang sangat buruk",
    "pelayanan buruk pengembalian dana dipersulit kapok beli di tempat ini"
]

adverbs = ["sungguh", "benar-benar", "amat", "sangat", "cukup", "pastinya", "tentu", "memang"]
connectors = ["dan juga", "namun", "bahkan", "sehingga", "karena itu", "selain itu", "maka"]

def generate_synthetic_reviews(n_samples=3000):
    texts = []
    labels = []
    for _ in range(n_samples // 2):
        # Positive review
        t1 = random.choice(pos_templates)
        t2 = random.choice(pos_templates)
        adv = random.choice(adverbs)
        conn = random.choice(connectors)
        review = f"{t1} {conn} {adv} {t2}"
        texts.append(review)
        labels.append(1)
        
        # Negative review
        t1 = random.choice(neg_templates)
        t2 = random.choice(neg_templates)
        adv = random.choice(adverbs)
        conn = random.choice(connectors)
        review = f"{t1} {conn} {adv} {t2}"
        texts.append(review)
        labels.append(0)
        
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)

print("[INFO-RNN/GRU] Menghasilkan korpus data ulasan teks sentimen...")
raw_texts, raw_labels = generate_synthetic_reviews(n_samples=3000)

# ---------------------------------------------------------------------------
# 3. Tokenisasi, Kosakata (Vocab), & PyTorch Dataset
# ---------------------------------------------------------------------------
def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = text.split()
    return tokens

# Bangun Vocabulary
word_counts = {}
for text in raw_texts:
    for token in clean_and_tokenize(text):
        word_counts[token] = word_counts.get(token, 0) + 1

vocab = {"<PAD>": 0, "<UNK>": 1}
for word, count in word_counts.items():
    if count >= 2:
        vocab[word] = len(vocab)

VOCAB_SIZE = len(vocab)
MAX_SEQ_LEN = 40
print(f"[INFO-RNN/GRU] Ukuran Kosakata (Vocab Size): {VOCAB_SIZE} kata, Panjang Sekuen Maksimal: {MAX_SEQ_LEN}")

def text_to_indices(text, max_len=MAX_SEQ_LEN):
    tokens = clean_and_tokenize(text)
    indices = [vocab.get(t, vocab["<UNK>"]) for t in tokens][:max_len]
    if len(indices) < max_len:
        indices += [vocab["<PAD>"]] * (max_len - len(indices))
    return indices

class TextSentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.data = [torch.tensor(text_to_indices(t), dtype=torch.long) for t in texts]
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Train-Test Split (80% Train, 20% Test)
split_idx = int(0.8 * len(raw_texts))
train_texts, test_texts = raw_texts[:split_idx], raw_texts[split_idx:]
train_labels, test_labels = raw_labels[:split_idx], raw_labels[split_idx:]

train_dataset = TextSentimentDataset(train_texts, train_labels)
test_dataset = TextSentimentDataset(test_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ---------------------------------------------------------------------------
# 4. Definisi Model Arsitektur: SimpleRNN vs GRU
# ---------------------------------------------------------------------------
class SimpleRNNModel(nn.Module):
    """
    Arsitektur Recurrent Neural Network Konvensional (Elman RNN).
    Rentan terhadap Vanishing / Exploding Gradient pada sekuen panjang.
    """
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, num_classes=2):
        super(SimpleRNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, nonlinearity='tanh')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        out, h_n = self.rnn(embedded)
        # Ambil hidden state terakhir
        out = self.dropout(h_n.squeeze(0))
        logits = self.fc(out)
        return logits

class GRUModel(nn.Module):
    """
    Arsitektur Gated Recurrent Unit (GRU).
    Mengatasi Vanishing Gradient dengan mekanisme Update Gate dan Reset Gate.
    """
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, num_classes=2):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        out, h_n = self.gru(embedded)
        out = self.dropout(h_n.squeeze(0))
        logits = self.fc(out)
        return logits

# ---------------------------------------------------------------------------
# 5. Fungsi Pelatihan dan Evaluasi
# ---------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * x.size(0)
        _, preds = torch.max(out, 1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total

def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            running_loss += loss.item() * x.size(0)
            _, preds = torch.max(out, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)

# ---------------------------------------------------------------------------
# 6. Pelatihan Komparatif SimpleRNN vs GRU
# ---------------------------------------------------------------------------
EPOCHS_RNN = 6
criterion = nn.CrossEntropyLoss()

# Pelatihan Simple RNN
print("\n" + "="*60)
print("MEMULAI PELATIHAN: MODEL 1 - SIMPLE RNN")
print("="*60)
rnn_model = SimpleRNNModel(VOCAB_SIZE).to(DEVICE)
opt_rnn = optim.Adam(rnn_model.parameters(), lr=0.002)
hist_rnn = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

t0_rnn = time.time()
best_acc_rnn = 0.0
for epoch in range(1, EPOCHS_RNN + 1):
    tr_loss, tr_acc = train_epoch(rnn_model, train_loader, criterion, opt_rnn, DEVICE)
    vl_loss, vl_acc, _, _ = eval_model(rnn_model, test_loader, criterion, DEVICE)
    hist_rnn['train_loss'].append(tr_loss)
    hist_rnn['train_acc'].append(tr_acc)
    hist_rnn['val_loss'].append(vl_loss)
    hist_rnn['val_acc'].append(vl_acc)
    if vl_acc > best_acc_rnn:
        best_acc_rnn = vl_acc
        torch.save(rnn_model.state_dict(), os.path.join(MODEL_DIR, "best_simple_rnn.pth"))
    print(f"Epoch [{epoch:02d}/{EPOCHS_RNN:02d}] | Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | Val Loss: {vl_loss:.4f} Acc: {vl_acc*100:.2f}%")
total_time_rnn = time.time() - t0_rnn

# Pelatihan GRU
print("\n" + "="*60)
print("MEMULAI PELATIHAN: MODEL 2 - GRU (GATED RECURRENT UNIT)")
print("="*60)
gru_model = GRUModel(VOCAB_SIZE).to(DEVICE)
opt_gru = optim.Adam(gru_model.parameters(), lr=0.002)
hist_gru = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

t0_gru = time.time()
best_acc_gru = 0.0
for epoch in range(1, EPOCHS_RNN + 1):
    tr_loss, tr_acc = train_epoch(gru_model, train_loader, criterion, opt_gru, DEVICE)
    vl_loss, vl_acc, _, _ = eval_model(gru_model, test_loader, criterion, DEVICE)
    hist_gru['train_loss'].append(tr_loss)
    hist_gru['train_acc'].append(tr_acc)
    hist_gru['val_loss'].append(vl_loss)
    hist_gru['val_acc'].append(vl_acc)
    if vl_acc > best_acc_gru:
        best_acc_gru = vl_acc
        torch.save(gru_model.state_dict(), os.path.join(MODEL_DIR, "best_gru.pth"))
    print(f"Epoch [{epoch:02d}/{EPOCHS_RNN:02d}] | Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | Val Loss: {vl_loss:.4f} Acc: {vl_acc*100:.2f}%")
total_time_gru = time.time() - t0_gru

# ---------------------------------------------------------------------------
# 7. Evaluasi Akhir & Visualisasi
# ---------------------------------------------------------------------------
rnn_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_simple_rnn.pth")))
gru_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_gru.pth")))

_, acc_rnn, preds_rnn, y_test_rnn = eval_model(rnn_model, test_loader, criterion, DEVICE)
_, acc_gru, preds_gru, y_test_gru = eval_model(gru_model, test_loader, criterion, DEVICE)

prec_rnn, rec_rnn, f1_rnn, _ = precision_recall_fscore_support(y_test_rnn, preds_rnn, average='weighted')
prec_gru, rec_gru, f1_gru, _ = precision_recall_fscore_support(y_test_gru, preds_gru, average='weighted')

print("\n--- HASIL EVALUASI SIMPLE RNN ---")
print(f"Akurasi: {acc_rnn*100:.2f}% | Precision: {prec_rnn*100:.2f}% | Recall: {rec_rnn*100:.2f}% | F1: {f1_rnn*100:.2f}%")

print("\n--- HASIL EVALUASI GRU ---")
print(f"Akurasi: {acc_gru*100:.2f}% | Precision: {prec_gru*100:.2f}% | Recall: {rec_gru*100:.2f}% | F1: {f1_gru*100:.2f}%")

# Simpan JSON
rnn_gru_results = {
    "task": "Sentiment Analysis Classification (NLP)",
    "vocab_size": VOCAB_SIZE,
    "max_seq_len": MAX_SEQ_LEN,
    "epochs": EPOCHS_RNN,
    "simple_rnn": {
        "training_time_sec": round(total_time_rnn, 2),
        "test_accuracy": round(float(acc_rnn) * 100, 2),
        "precision": round(float(prec_rnn) * 100, 2),
        "recall": round(float(rec_rnn) * 100, 2),
        "f1_score": round(float(f1_rnn) * 100, 2),
        "history": hist_rnn
    },
    "gru": {
        "training_time_sec": round(total_time_gru, 2),
        "test_accuracy": round(float(acc_gru) * 100, 2),
        "precision": round(float(prec_gru) * 100, 2),
        "recall": round(float(rec_gru) * 100, 2),
        "f1_score": round(float(f1_gru) * 100, 2),
        "history": hist_gru
    }
}
with open(os.path.join(OUTPUT_DIR, "rnn_gru_results.json"), "w") as f:
    json.dump(rnn_gru_results, f, indent=4)

# Visualisasi Kurva
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS_RNN + 1), hist_rnn['train_loss'], 'b--', label='RNN Train Loss')
plt.plot(range(1, EPOCHS_RNN + 1), hist_rnn['val_loss'], 'b-', label='RNN Val Loss')
plt.plot(range(1, EPOCHS_RNN + 1), hist_gru['train_loss'], 'g--', label='GRU Train Loss')
plt.plot(range(1, EPOCHS_RNN + 1), hist_gru['val_loss'], 'g-', label='GRU Val Loss')
plt.title("Perbandingan Kurva Loss: SimpleRNN vs GRU", fontsize=12, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS_RNN + 1), [x*100 for x in hist_rnn['train_acc']], 'b--', label='RNN Train Acc')
plt.plot(range(1, EPOCHS_RNN + 1), [x*100 for x in hist_rnn['val_acc']], 'b-', label='RNN Val Acc')
plt.plot(range(1, EPOCHS_RNN + 1), [x*100 for x in hist_gru['train_acc']], 'g--', label='GRU Train Acc')
plt.plot(range(1, EPOCHS_RNN + 1), [x*100 for x in hist_gru['val_acc']], 'g-', label='GRU Val Acc')
plt.title("Perbandingan Kurva Akurasi: SimpleRNN vs GRU", fontsize=12, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Akurasi (%)")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rnn_gru_loss_accuracy.png"), dpi=200)
plt.close()

# Visualisasi Barchart
plt.figure(figsize=(8, 5))
labels_cat = ['Akurasi (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)']
rnn_s = [acc_rnn*100, prec_rnn*100, rec_rnn*100, f1_rnn*100]
gru_s = [acc_gru*100, prec_gru*100, rec_gru*100, f1_gru*100]
x = np.arange(len(labels_cat))
w = 0.35
plt.bar(x - w/2, rnn_s, w, label='Simple RNN', color='#8e44ad')
plt.bar(x + w/2, gru_s, w, label='GRU (Gated Recurrent Unit)', color='#27ae60')
plt.xticks(x, labels_cat)
plt.ylim(0, 105)
plt.ylabel("Persentase (%)")
plt.title("Perbandingan Metrik NLP: SimpleRNN vs GRU", fontsize=12, fontweight='bold')
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rnn_gru_metrics_barchart.png"), dpi=200)
plt.close()

print(f"[SUKSES-RNN/GRU] Pelatihan dan evaluasi RNN vs GRU selesai, visualisasi tersimpan di {OUTPUT_DIR}/")
