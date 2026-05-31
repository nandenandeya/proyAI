# 🎯 FocusWebCam — Deteksi Fokus Wajah dengan AI

Aplikasi deteksi tingkat fokus secara **real-time** menggunakan webcam, dibangun dengan
**MediaPipe FaceLandmarker** dan **Logistic Regression**.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 🎯 Fitur Utama

| Fitur | Deskripsi |
|---|---|
| 🎥 Live Detection | Deteksi fokus real-time dari webcam via WebRTC |
| 👁️ EAR | Eye Aspect Ratio — mengukur seberapa terbuka mata |
| 🔄 Head Pose | Estimasi posisi kepala terhadap kamera |
| 👄 Mouth Ratio | Rasio bukaan mulut (mendeteksi menguap) |
| 📊 Session Report | Grafik perkembangan fokus per sesi |
| 💡 Explainable AI | Penjelasan alasan di balik setiap prediksi |

---

## 📚 Konsep Machine Learning

| Konsep | Implementasi |
|---|---|
| Supervised Learning | Model dilatih dengan data berlabel (fokus / tidak fokus) |
| Feature Extraction | EAR, Head Pose, Mouth Ratio dari 478 landmark wajah |
| Logistic Regression | Klasifikasi biner dengan output probabilitas |
| Standardization | Normalisasi fitur `(x−mean)/std` |
| Smoothing | Window rata-rata 5 frame untuk prediksi stabil |
| Explainable AI | Kontribusi tiap fitur dijelaskan ke pengguna |

---

## 🛠️ Stack Teknologi

- **Streamlit 1.57** — framework web
- **streamlit-webrtc 0.47** — webcam stream di cloud
- **MediaPipe 0.10.9** — deteksi landmark wajah (API FaceLandmarker baru)
- **OpenCV 4.8 (headless)** — pemrosesan gambar
- **NumPy 1.26 / Pandas / Plotly** — numerik & visualisasi

---

## 🚀 Menjalankan Secara Lokal

### Prasyarat
- Python 3.9 – 3.11 (disarankan 3.10)
- pip

### Langkah-langkah

```bash
# 1. Clone / download proyek
git clone https://github.com/<username>/focuswebcam.git
cd focuswebcam

# 2. Buat virtual environment
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

# 3. Install dependensi Python
pip install -r requirements.txt

# 4. (Linux) Install paket sistem jika belum ada
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# 5. Jalankan aplikasi
streamlit run app.py
```

Buka browser ke **http://localhost:8501**

### Troubleshooting Lokal

| Error | Solusi |
|---|---|
| `ImportError: libGL.so` | `sudo apt-get install libgl1-mesa-glx` |
| `ModuleNotFoundError: mediapipe` | `pip install mediapipe==0.10.9` |
| Webcam tidak muncul | Coba browser Chrome/Edge, izinkan kamera |
| `av` install gagal | `pip install av==10.0.0 --no-binary av` |

---

## ☁️ Deploy ke Streamlit Community Cloud

### 1. Siapkan Repository GitHub

```bash
git init
git add .
git commit -m "feat: initial FocusWebCam deployment"
git branch -M main
git remote add origin https://github.com/<username>/focuswebcam.git
git push -u origin main
```

### 2. Deploy

1. Buka [share.streamlit.io](https://share.streamlit.io)
2. Klik **"New app"**
3. Pilih repository `focuswebcam`
4. Branch: `main`
5. Main file path: `app.py`
6. Klik **"Deploy!"**

### 3. Melihat Log Error

Di dashboard Streamlit Cloud → klik app → **"Manage app"** → **"Logs"**

### 4. Update Aplikasi

```bash
git add .
git commit -m "fix: update changes"
git push
```
Streamlit Cloud otomatis re-deploy setelah push.

---

## 📁 Struktur Proyek

```
focuswebcam/
├── app.py                  # Aplikasi utama Streamlit
├── requirements.txt        # Dependensi Python
├── packages.txt            # Paket sistem Linux (Streamlit Cloud)
├── features.csv            # Dataset fitur (untuk referensi/training ulang)
├── .streamlit/
│   └── config.toml         # Konfigurasi tema & server
└── README.md
```

> **Catatan:** File `face_landmarker.task` (~10 MB) **tidak perlu** di-commit ke Git.
> Aplikasi mengunduhnya otomatis dari Google Storage saat pertama kali dijalankan.
> Tambahkan ke `.gitignore`.

---

## 📈 Performa Model

| Metrik | Nilai |
|---|---|
| Accuracy | 81.45% |
| F1 Score | 0.834 |
| Precision (FOKUS) | 0.79 |
| Recall (FOKUS) | 0.88 |

---

## ⚠️ Catatan Penggunaan

- Aplikasi membutuhkan izin akses kamera dari browser
- Gunakan **Chrome** atau **Edge** untuk hasil terbaik
- WebRTC membutuhkan koneksi HTTPS — otomatis terpenuhi di Streamlit Cloud
- Model MediaPipe diunduh sekali (~10 MB) dan di-cache
