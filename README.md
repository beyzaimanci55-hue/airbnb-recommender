# 🏡 Airbnb Hybrid Recommendation System

A multi-layer recommendation engine for Airbnb listings built with Python, scikit-learn, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Overview

This project implements a **hybrid recommendation system** that combines three algorithmic layers to suggest the best Airbnb listings based on user preferences:

| Layer | Method | What it does |
|-------|--------|--------------|
| 1 | **Association-Rule Inspired** | Identifies the most popular city for a given season |
| 2 | **Content-Based Filtering** | Finds listings similar to the top-quality seed via cosine similarity |
| 3 | **Quality-Enhanced Scoring** | Blends 7 weighted review dimensions into a quality score |

The final **Hybrid Score** is a configurable weighted combination of content similarity and quality score — adjustable live in the UI.

---

## 🖥 Demo

![Demo screenshot placeholder](https://via.placeholder.com/900x500?text=Streamlit+Demo+Screenshot)

### Features
- 🔍 **Filter** by city, season, property type, price range, beds, superhost status
- ⚖️ **Tune** the content vs. quality weight slider in real-time
- 🗺 **Interactive map** of recommended listings (Plotly + OpenStreetMap)
- 📊 **Radar chart** comparing recommendation scores vs. city average
- 📋 **Score breakdown** showing how each component contributes

---

## 📁 Project Structure

```
airbnb-recommender/
├── app.py                  # Ana uygulama (Streamlit)
├── recommender.py          # Hibrit öneri motoru mantığı
├── user_profile.py         # Kullanıcı profili ve öğrenme sistemi
├── date_insights.py        # Tarih ve sezon analitiği
├── airbnb.csv              # Veri seti
├── requirements.txt        # Gerekli kütüphaneler
└── README.md
```

---

## 🚀 Başlangıç

### 1. Projeyi İndirin
```bash
git clone https://github.com/<kullanici-adiniz>/airbnb-recommender.git
cd airbnb-recommender
```

### 2. Bağımlılıkları Kurun
```bash
pip install -r requirements.txt
```

### 3. Veri Seti
`airbnb.csv` dosyasının ana dizinde olduğundan emin olun.

### 4. Run the app
```bash
streamlit run app.py
```

---

## 🌐 Canlıya Taşıma (Sitenizi Paylaşın)

Bu projeyi başkalarıyla paylaşmanın en kolay yolu **Streamlit Community Cloud** kullanmaktır:

1. **Kodunuzu GitHub'a Yükleyin:**
   - GitHub'da yeni bir repo oluşturun.
   - Yerel kodunuzu oraya gönderin:
     ```bash
     git init
     git add .
     git commit -m "İlk sürüm"
     git branch -M main
     git remote add origin https://github.com/<kullanici-adiniz>/<repo-adiniz>.git
     git push -u origin main
     ```

2. **Streamlit Cloud'da Yayınlayın:**
   - [share.streamlit.io](https://share.streamlit.io/) adresine gidin.
   - GitHub hesabınızı bağlayın.
   - "New app" butonuna basın, reponuzu, dalı (`main`) ve ana dosyayı (`app.py`) seçin.
   - **Deploy!** butonuna basın. Siteniz birkaç dakika içinde canlıda olacaktır.

3. **Veri Yönetimi:**
   - `airbnb.csv` dosyasının GitHub'da olduğundan emin olun (dosya boyutu büyükse Git LFS gerekebilir).

---

## 🧠 Model Nasıl Çalışır?

### Content-Based Filtering
Property features (price, beds, bedrooms, bathrooms, guest capacity, property type, neighbourhood) are one-hot encoded, normalized with `MinMaxScaler`, and compared with **cosine similarity**. The highest-quality listing in the filtered pool acts as the seed.

### Quality Score
Seven review dimensions are weighted and combined into a single quality score:

| Dimension | Weight |
|-----------|--------|
| Rating score | 25% |
| Cleanliness score | 20% |
| Location score | 20% |
| Value for money score | 15% |
| Accuracy score | 10% |
| Checkin score | 5% |
| Communication score | 5% |

### Hybrid Ranking
```
hybrid_score = w_content × content_score + w_quality × quality_score
```
Both scores are normalized to [0,1] before combining. The user controls `w_content` via a slider (default 0.5 / 0.5).

---

## 📊 Dataset

- **Source:** Airbnb listings across 5 Italian cities (Roma, Milano, Firenze, Napoli, Venezia)
- **Size:** 282,047 listings × 26 features
- **Coverage:** 4 seasons — Early Spring, Early Summer, Early Autumn, Early Winter
- The app samples 12,000 rows by default for performance; adjust `sample_size` in `load_data()`.

---

## 🛠 Tech Stack

- **Python 3.11+**
- **pandas / numpy** — data processing
- **scikit-learn** — cosine similarity, normalization
- **Streamlit** — interactive web UI
- **Plotly** — charts and maps

---

## 📌 Possible Improvements

- [ ] Add collaborative filtering (SVD / ALS) when user-interaction data is available
- [ ] Cache similarity matrix to disk for faster cold start
- [ ] Deploy to Streamlit Community Cloud
- [ ] Add unit tests (`pytest`)
- [ ] Neighbourhood-level heatmaps

---

## 👤 Author

**Beyza İmancı**  
Computer Engineering · CENG468 — Recommendation Systems Project

---

## 📄 License

MIT — feel free to fork and build on this.
