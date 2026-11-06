# 🎓 Tugas Mandiri Praktikum Deep Learning: Implementasi & Komparasi Arsitektur

Repositori ini berisi solusi lengkap, siap pakai, dan terstandarisasi untuk **Tugas Mandiri Deep Learning (Semester 5)**. Proyek ini mengimplementasikan, melatih, mengevaluasi, dan membandingkan secara komparatif arsitektur Deep Learning sesuai ketentuan tugas.

---

## 📌 Ringkasan Pilihan Model & Dataset

| Komponen | Pilihan Utama (Computer Vision - Direkomendasikan) | Jalur Alternatif (NLP / Pemodelan Sekuensial) |
| :--- | :--- | :--- |
| **Arsitektur 1** | **Custom Deep CNN** (5 Lapisan Konvolusi dari Scratch + BatchNorm + Dropout) | **SimpleRNN** (Elman Network) |
| **Arsitektur 2** | **Transfer Learning (ResNet-18)** (Pre-trained ImageNet + Fine-Tuning) | **GRU** (Gated Recurrent Unit) |
| **Dataset** | **Fashion-MNIST** (10 Kelas Pakaian & Aksesori Mode) | **Sentiment Analysis Review Corpus** (3.000 Sampel Teks) |
| **Kerangka Kerja** | PyTorch & Torchvision | PyTorch |
| **Metrik Evaluasi** | Akurasi, Weighted Precision, Recall, F1-Score, Confusion Matrix | Akurasi, Precision, Recall, F1-Score |

---

## 📁 Struktur Direktori Proyek

```text
t1prof/
├── JAWABAN_FORM_SUBMISSION.md        <-- Template isian jawaban formulir Google Form / LMS
├── LAPORAN_TUGAS_DEEP_LEARNING.md    <-- Laporan akademik formal lengkap (Bab I - V)
├── README.md                         <-- Panduan penggunaan proyek ini
│
├── notebook_cnn_transfer_learning.ipynb  <-- Jupyter Notebook utama (Vision: CNN vs Transfer Learning)
├── notebook_rnn_gru_sentiment.ipynb      <-- Jupyter Notebook suplemen (NLP: SimpleRNN vs GRU)
│
├── train_vision_models.py            <-- Script eksekusi pelatihan CNN & ResNet-18
├── train_rnn_gru.py                  <-- Script eksekusi pelatihan SimpleRNN & GRU
├── generate_notebooks.py             <-- Script pembangun Jupyter Notebook
│
├── data_fmnist/                      <-- Dataset Fashion-MNIST (Auto-downloaded)
├── models/                           <-- Checkpoint bobot terbaik (.pth)
│   ├── best_custom_cnn.pth
│   ├── best_transfer_learning.pth
│   ├── best_simple_rnn.pth
│   └── best_gru.pth
│
└── outputs/                          <-- Grafik visualisasi & hasil evaluasi
    ├── fashion_mnist_samples.png
    ├── loss_accuracy_comparison.png
    ├── confusion_matrix_cnn.png
    ├── confusion_matrix_transfer_learning.png
    ├── metrics_comparison_barchart.png
    └── evaluation_results.json
```

---

## 💻 Cara Menjalankan Secara Lokal (Local Environment)

### 1. Instalasi Dependensi
Pastikan Python 3.9+ telah terpasang. Jalankan instalasi dependensi berikut:
```bash
pip install torch torchvision scikit-learn matplotlib seaborn nbformat
```

### 2. Menjalankan Pelatihan Computer Vision (CNN vs ResNet-18):
```bash
python train_vision_models.py
```
Script akan melatih kedua arsitektur, mengevaluasi data uji, menyimpan model terbaik ke folder `models/`, dan menghasilkan seluruh grafik ke folder `outputs/`.

### 3. Menjalankan Pelatihan NLP / Sekuensial (SimpleRNN vs GRU):
```bash
python train_rnn_gru.py
```

### 4. Membuka Jupyter Notebook:
```bash
jupyter notebook notebook_cnn_transfer_learning.ipynb
```

---

## ☁️ Cara Menjalankan di Google Colab

1. Buka [Google Colab](https://colab.research.google.com/).
2. Pilih tab **Upload** dan pilih file `notebook_cnn_transfer_learning.ipynb`.
3. Aktifkan akselerasi GPU: Klik menu **Runtime** > **Change runtime type** > Pilih **T4 GPU** > Klik **Save**.
4. Klik **Runtime** > **Run all** (Seluruh sel akan dieksekusi dengan cepat memanfaatkan GPU).

---

## 📊 Ringkasan Hasil Eksperimen Utama (Fashion-MNIST)

| Parameter Evaluasi | Custom CNN (Scratch) | Transfer Learning (ResNet-18) | Catatan Analisis |
| :--- | :---: | :---: | :--- |
| **Akurasi Pengujian** | **88.25%** | **91.95%** | **ResNet-18 unggul +3.70%** berkat transfer fitur ImageNet |
| **Weighted F1-Score** | 88.15% | **91.95%** | Keseimbangan presisi dan sensitivitas sangat stabil |
| **Jumlah Parameter** | **437.738** | 11.310.410 | **Custom CNN 25x lebih hemat memori** |
| **Waktu Pelatihan** | **Cepat & Ringkas** | Sedikit lebih berat | Custom CNN sangat ideal untuk perangkat berdaya rendah |
