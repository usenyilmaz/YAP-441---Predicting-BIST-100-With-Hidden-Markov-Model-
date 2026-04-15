# BIST-100 Yön Tahmini — Saklı Markov Modeli

**Yazar:** Ural Şenyılmaz — 241401014  
**Ders:** TOBB ETÜ YAP 441 / BİL 541 Dönem Projesi — 2025-26 Bahar  
**Konu:** Makroekonomik veriler ve teknik göstergeler kullanılarak Saklı Markov Modeli (HMM) ile haftalık BIST-100 endeksinin piyasa rejimini tespit etmek ve bu rejim sinyallerine dayalı bir al-sat stratejisi geliştirmek.

---

## Proje Özeti

Bu proje iki aşamalı bir yaklaşım izlemektedir:

1. **İkili Yön Tahmini (Başarısız):** HMM ile haftalık fiyatın artış mı azalış mı göstereceği tahmin edilmeye çalışılmış; modelin eğitim döneminin baskın yükseliş eğilimini içselleştirmesi nedeniyle başarısız olduğu deneysel olarak gösterilmiştir.

2. **Rejim Tespiti (Başarılı):** HMM, piyasayı Boğa / Yatay / Ayı rejimlerine ayırmak için yeniden yapılandırılmıştır. Boğa rejiminde pozisyon tutulmuş, diğer rejimlerde nakite geçilmiştir. 2022–2024 test döneminde strateji, buy-and-hold'u getiri (%457.3 vs %432.4), maksimum drawdown (%-20.1 vs %-21.2) ve Sharpe oranı (2.10 vs 2.03) açısından geride bırakmıştır.

---

## Dosya Yapısı

```
├── data_collection.py          # Veri toplama ve ön işleme
├── hmm_upward_downward.py      # İkili yön tahmini (başarısız yaklaşım)
├── hmm_regime.py               # Rejim tespiti ve al-sat stratejisi
├── visualize.py                # Grafik üretimi (4 görsel)
│
├── data/
│   ├── bist100_raw.csv         # Ham veri (734 hafta, 26 sütun)
│   ├── bist100_normalized.csv  # Min-Max normalize edilmiş veri
│   ├── backtest_results.csv    # İkili tahmin backtest sonuçları
│   └── regime_backtest_results.csv  # Rejim stratejisi sonuçları
│
├── figures/
│   ├── fig1_transition_matrix.png   # HMM geçiş matrisi ısı haritası
│   ├── fig2_cumulative_returns.png  # Kümülatif getiri karşılaştırması
│   ├── fig4_regime_distribution.png # Eğitim verisi rejim dağılımı
│
└── bist100_hmm_rapor.tex       # IEEE formatında LaTeX raporu
```

---

## Kurulum

Python 3.12 gereklidir. Tüm bağımlılıkları kurmak için:

```bash
py -3.12 -m pip install yfinance pandas-ta hmmlearn scikit-learn matplotlib requests
```

---

## Çalıştırma Sırası

### 1. Veri Toplama
```bash
py -3.12 data_collection.py
```
`data/` klasörüne `bist100_raw.csv` ve `bist100_normalized.csv` dosyalarını oluşturur. TCMB API anahtarı girilmemişse faiz/TÜFE/ÜFE sütunları atlanır.

### 2. İkili Yön Tahmini (Opsiyonel — Başarısız Yaklaşım)
```bash
py -3.12 hmm_upward_downward.py
```
Fiyat yönü tahmini için HMM dener; yakınsama sorununu ve başarısız sonuçları raporlar.

### 3. Rejim Tespiti ve Al-Sat Stratejisi
```bash
py -3.12 hmm_regime.py
```
N=3 GaussianHMM ile Boğa/Yatay/Ayı rejimlerini tespit eder, walk-forward backtesting uygular ve performans metriklerini raporlar.

### 4. Görselleştirme
```bash
py -3.12 visualize.py
```
`figures/` klasörüne 4 grafik dosyası üretir.

---

## Kullanılan Veri Kaynakları

| Veri | Kaynak | Sembol |
|------|--------|--------|
| BIST-100 fiyat | Yahoo Finance | `XU100.IS` |
| USD/TRY kuru | Yahoo Finance | `TRY=X` |
| Brent ham petrol | Yahoo Finance | `BZ=F` |
| S&P 500 | Yahoo Finance | `^GSPC` |
| DAX | Yahoo Finance | `^GDAXI` |
| TCMB faiz / TÜFE / ÜFE | TCMB EVDS API | — (API kısıtlaması) |

**Kapsam:** Ocak 2010 – Aralık 2024 | **Frekans:** Haftalık | **Gözlem:** 734

---

## Teknik Göstergeler

`pandas-ta` kütüphanesi ile hesaplanmıştır:

- **RSI** (14 periyot)
- **MACD** (12-26-9) + sinyal çizgisi + histogram
- **SMA-20, SMA-50, EMA-20**
- **Bollinger Bantları** (20 periyot, 2 std) — üst/alt/orta bant + bant genişliği
- **Normalize Hacim** (20 haftalık ortalamaya göre)

---

## Model Sonuçları

| Strateji | Getiri | Max. Drawdown | Sharpe |
|----------|--------|---------------|--------|
| **HMM Rejim Stratejisi** | **%457.3** | **%-20.1** | **2.10** |
| Buy-and-Hold | %432.4 | %-21.2 | 2.03 |
| Rastgele Strateji | %145.6 | %-20.0 | 1.44 |

> **Not:** Getiriler TL bazlıdır ve 2022–2024 yüksek enflasyon dönemini kapsamaktadır. USD bazlı reel getiriler bu değerlerin önemli ölçüde altındadır.
