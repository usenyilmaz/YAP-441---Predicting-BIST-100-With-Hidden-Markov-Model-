"""
BIST-100 Yön Tahmini Projesi
Adım 2: HMM Modeli — Eğitim, Optimizasyon, Backtesting
========================================================
Yazar : Ural Şenyılmaz
Tarih : 2026

Raporda taahhüt edilenler:
  - GaussianHMM (hmmlearn)
  - Gizli durum sayısı: N=2, N=3, N=4 → BIC/AIC ile seçim
  - Baum-Welch eğitimi, çoklu random_state ile en iyi log-likelihood
  - Viterbi ile durum dizisi çıkarımı
  - Walk-forward backtesting (2022-2024)
  - Accuracy, Precision, Recall, F1, Confusion Matrix
  - Baseline karşılaştırması: rastgele tahmin + buy-and-hold
"""

import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.preprocessing import MinMaxScaler
import warnings
import os

warnings.filterwarnings("ignore")

# ── Ayarlar ────────────────────────────────────────────────────────────────────

DATA_PATH  = "data/bist100_raw.csv"
RAW_PATH   = "data/bist100_raw.csv"
OUTPUT_DIR = "data"

# Raporda belirlenen eğitim/test bölümü
TRAIN_END  = "2021-12-31"
TEST_START = "2022-01-01"

# HMM denenecek gizli durum sayıları
N_STATES_LIST = [2, 3, 4]

# Her N için kaç farklı başlangıç noktası denensin
N_INIT = 20

# Modele verilecek özellikler — 8 özellik (aşırı parametre sorununu önler)
# Ham seviye değerleri (usdtry, brent, sp500, dax) kaldırıldı: değişim oranları
# daha bilgilendirici. Yüksek korelasyonlu göstergeler (sma_20, sma_50, ema_20,
# bb_upper, bb_mid, bb_lower) kaldırıldı: bb_width zaten volatiliteyi özetliyor.
FEATURE_COLS = [
    # Teknik göstergeler
    "rsi", "macd", "bb_width", "volume_norm",
    # Makroekonomik değişkenler (değişim oranları — seviye değil)
    "usdtry_chg", "brent_chg", "sp500_chg", "dax_chg",
]


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 1 — Veri Yükle ve Böl
# ══════════════════════════════════════════════════════════════════════════════

def load_and_split():
    """
    Ham veriyi yükler, önce eğitim/test olarak böler, ardından normalize eder.
    MinMaxScaler yalnızca eğitim setine fit edilir; test setine sadece transform
    uygulanır — bu sayede test verisi normalizasyona sızmaz (data leakage önlenir).
    """
    print("📂 Veri yükleniyor...")

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    # Sadece kullanacağımız özellikler + hedef
    cols = FEATURE_COLS + ["target"]
    df = df[cols].dropna()

    train = df[df.index <= TRAIN_END]
    test  = df[df.index >= TEST_START]

    X_train = train[FEATURE_COLS].values
    y_train = train["target"].values
    X_test  = test[FEATURE_COLS].values
    y_test  = test["target"].values

    # Normalizasyon sızıntısını önle: scaler SADECE eğitim setine fit edilir,
    # test setine yalnızca transform uygulanır.
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"   ✓ Eğitim : {len(train)} hafta  ({train.index[0].date()} → {train.index[-1].date()})")
    print(f"   ✓ Test   : {len(test)}  hafta  ({test.index[0].date()} → {test.index[-1].date()})")
    print(f"   ✓ Özellik sayısı: {len(FEATURE_COLS)}")

    return X_train, y_train, X_test, y_test, train, test


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 2 — HMM Eğitimi (Baum-Welch, çoklu başlangıç)
# ══════════════════════════════════════════════════════════════════════════════

def train_hmm(X_train, n_states, n_init=N_INIT):
    """
    Verilen gizli durum sayısı için GaussianHMM eğitir.
    Baum-Welch (EM) algoritması hmmlearn tarafından otomatik kullanılır.
    Farklı random_state'lerle n_init kez dener, en yüksek log-likelihood'u seçer.
    """
    best_model  = None
    best_score  = -np.inf

    for seed in range(n_init):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",   # köşegen kovaryans: d parametre/durum (full=d*(d+1)/2)
            n_iter=500,               # maksimum EM iterasyonu (yakınsama için artırıldı)
            tol=1e-4,                 # yakınsama toleransı
            random_state=seed,
        )
        try:
            model.fit(X_train)
            score = model.score(X_train)   # log-likelihood
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    return best_model, best_score


def select_best_n(X_train, n_states_list=N_STATES_LIST):
    """
    N=2, 3, 4 için modelleri eğitir.
    BIC ve AIC hesaplayarak en iyi N'i seçer.
    BIC = -2*logL + k*ln(n)   (k = parametre sayısı)
    AIC = -2*logL + 2*k
    """
    print("\n🔍 Gizli durum sayısı optimizasyonu (N=2,3,4)...")
    print(f"   Her N için {N_INIT} farklı başlangıç noktası deneniyor...\n")

    n_samples, n_features = X_train.shape
    results = []

    for n in n_states_list:
        model, log_likelihood = train_hmm(X_train, n_states=n)

        if model is None:
            print(f"   N={n} → eğitim başarısız")
            continue

        # Parametre sayısı: geçiş matrisi + emisyon parametreleri
        # diag covariance: n*(n-1) geçiş + n*d ortalama + n*d kovaryans (köşegen)
        d = n_features
        k_params = (n * (n - 1)) + (n * d) + (n * d)

        bic = -2 * log_likelihood + k_params * np.log(n_samples)
        aic = -2 * log_likelihood + 2 * k_params

        results.append({
            "n_states"       : n,
            "log_likelihood" : round(log_likelihood, 2),
            "bic"            : round(bic, 2),
            "aic"            : round(aic, 2),
            "model"          : model,
        })

        print(f"   N={n} | LogL: {log_likelihood:>12.2f} | BIC: {bic:>12.2f} | AIC: {aic:>12.2f}")

    # En düşük BIC = en iyi model
    best = min(results, key=lambda x: x["bic"])
    print(f"\n   ✅ Seçilen model: N={best['n_states']} (en düşük BIC)")

    return best["model"], best["n_states"], results


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 3 — Gizli Durumları Yorumla (Viterbi)
# ══════════════════════════════════════════════════════════════════════════════

def interpret_states(model, X_train, y_train, n_states):
    """
    Viterbi algoritmasıyla eğitim verisindeki gizli durum dizisini çıkarır.
    Her durumun ortalama hedef etiketine bakarak rejim yorumu yapar:
      - Ortalaması yüksek durum → "Boğa" (artış ağırlıklı)
      - Ortalaması düşük durum  → "Ayı"  (düşüş ağırlıklı)
      - Ortası                  → "Yatay"
    """
    hidden_states = model.predict(X_train)   # Viterbi

    state_info = {}
    for s in range(n_states):
        mask = hidden_states == s
        avg_return = y_train[mask].mean() if mask.sum() > 0 else 0.5
        state_info[s] = {
            "avg_target" : round(avg_return, 3),
            "count"      : int(mask.sum()),
        }

    # Durumları artış ortalamasına göre sırala → rejim ismi ver
    sorted_states = sorted(state_info.items(), key=lambda x: x[1]["avg_target"])
    regime_names  = ["Ayı (Düşüş)", "Yatay", "Boğa (Yükseliş)"]

    print(f"\n📊 Gizli Durum Yorumu (Viterbi — eğitim verisi):")
    for rank, (state_id, info) in enumerate(sorted_states):
        regime = regime_names[rank] if n_states == 3 else (
            "Ayı" if rank == 0 else "Boğa"
        )
        print(f"   Durum {state_id}: {regime:<20} | "
              f"Artış oranı: %{info['avg_target']*100:.1f} | "
              f"Hafta sayısı: {info['count']}")

    # Hangi durum → artış (1), hangi durum → düşüş (0)
    # En yüksek avg_target'lı durum(lar) → tahmin=1
    sorted_by_return = sorted(state_info.items(),
                               key=lambda x: x[1]["avg_target"], reverse=True)
    bullish_states = {s for s, _ in sorted_by_return[:n_states//2 + n_states%2]}

    return hidden_states, bullish_states, state_info


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 4 — Walk-Forward Backtesting
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward_backtest(model, X_train, bullish_states, X_test, y_test):
    """
    Walk-forward backtesting:
    Her adımda model X_train + o ana kadarki test verisini görür,
    bir sonraki haftayı tahmin eder.
    Bu yöntem data leakage'ı önler — model geleceği görmez.
    """
    print("\n⏱️  Walk-forward backtesting çalışıyor...")

    predictions = []
    probabilities = []

    X_so_far = X_train.copy()

    for i in range(len(X_test)):
        # Mevcut tüm veriyle durum dizisini çıkar
        hidden = model.predict(X_so_far)
        current_state = hidden[-1]   # en son durum

        # Geçiş olasılıklarına göre bir sonraki durum tahmini
        next_state_probs = model.transmat_[current_state]

        # Boğa durumlarının toplam olasılığı = artış ihtimali
        bull_prob = sum(next_state_probs[s] for s in bullish_states)

        pred = 1 if bull_prob >= 0.5 else 0
        predictions.append(pred)
        probabilities.append(round(bull_prob, 4))

        # Bir sonraki adım için bu haftanın verisini ekle
        X_so_far = np.vstack([X_so_far, X_test[i]])

    predictions  = np.array(predictions)
    probabilities = np.array(probabilities)

    print(f"   ✓ {len(predictions)} haftalık tahmin tamamlandı")
    return predictions, probabilities


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 5 — Performans Değerlendirmesi
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(y_test, predictions, probabilities, test_df):
    """
    Raporda taahhüt edilen tüm metrikler:
    Accuracy, Precision, Recall, F1, Confusion Matrix
    + Baseline karşılaştırmaları
    """
    acc  = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, zero_division=0)
    rec  = recall_score(y_test, predictions, zero_division=0)
    f1   = f1_score(y_test, predictions, zero_division=0)
    cm   = confusion_matrix(y_test, predictions)

    print("\n" + "═"*60)
    print("📈 MODEL PERFORMANSI (Test: 2022-2024)")
    print("═"*60)
    print(f"  Accuracy  (Doğruluk)  : %{acc*100:.2f}   ← hedef: ≥%60")
    print(f"  Precision (Kesinlik)  : %{prec*100:.2f}")
    print(f"  Recall    (Duyarlılık): %{rec*100:.2f}")
    print(f"  F1 Skoru             : %{f1*100:.2f}")
    print()
    print("  Confusion Matrix:")
    print(f"  {'':>12} Tahmin:Düşüş  Tahmin:Artış")
    print(f"  Gerçek:Düşüş     {cm[0][0]:>5}          {cm[0][1]:>5}")
    print(f"  Gerçek:Artış     {cm[1][0]:>5}          {cm[1][1]:>5}")

    # ── Baseline 1: Rastgele Tahmin ──────────────────────────────────────────
    np.random.seed(42)
    random_preds = np.random.randint(0, 2, len(y_test))
    random_acc   = accuracy_score(y_test, random_preds)

    # ── Baseline 2: Her Zaman Artış Tahmin Et ────────────────────────────────
    always_up_acc = y_test.mean()

    print()
    print("  Baseline Karşılaştırması:")
    print(f"  Rastgele tahmin      : %{random_acc*100:.2f}")
    print(f"  Her zaman artış      : %{always_up_acc*100:.2f}")
    print(f"  HMM modelimiz        : %{acc*100:.2f}")

    if acc >= 0.60:
        print(f"\n  ✅ HEDEF AŞILDI! (%60 hedefinin üzerinde)")
    else:
        print(f"\n  ⚠️  Henüz hedefe ulaşılamadı (fark: %{(0.60-acc)*100:.2f})")

    # ── Baseline 3: Buy-and-Hold Karşılaştırması ─────────────────────────────
    raw = pd.read_csv(RAW_PATH, index_col=0, parse_dates=True)
    raw_test = raw[raw.index >= TEST_START]["close"]

    if len(raw_test) > 1:
        bh_return = (raw_test.iloc[-1] / raw_test.iloc[0] - 1) * 100

        # Sinyal tabanlı strateji: artış tahmini varsa tut, düşüş tahmini varsa çık
        weekly_returns = raw_test.pct_change().dropna().values
        min_len = min(len(predictions), len(weekly_returns))
        strategy_return = (weekly_returns[:min_len] * predictions[:min_len]).sum() * 100

        print()
        print("  Piyasa Simülasyonu (test dönemi):")
        print(f"  Buy-and-Hold getirisi : %{bh_return:.1f}")
        print(f"  HMM strateji getirisi : %{strategy_return:.1f}")
        print("  (Komisyon/vergi ihmal edilmiştir)")

    print("═"*60)

    # Sonuçları kaydet
    result_df = pd.DataFrame({
        "tarih"     : test_df.index[:len(predictions)],
        "gercek"    : y_test[:len(predictions)],
        "tahmin"    : predictions,
        "artis_olasiligi": probabilities,
    })
    result_df.to_csv(os.path.join(OUTPUT_DIR, "backtest_results.csv"), index=False)
    print(f"\n  ✅ Backtest sonuçları → data/backtest_results.csv")

    return acc, prec, rec, f1


# ══════════════════════════════════════════════════════════════════════════════
# ANA FONKSİYON
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═"*60)
    print("  BIST-100 YÖN TAHMİNİ — HMM MODELİ")
    print("═"*60)

    # 1. Veri yükle ve böl
    X_train, y_train, X_test, y_test, train_df, test_df = load_and_split()

    # 2. N=2,3,4 için modelleri eğit, BIC ile en iyisini seç
    best_model, best_n, all_results = select_best_n(X_train)

    # 3. Gizli durumları Viterbi ile yorumla
    hidden_states, bullish_states, state_info = interpret_states(
        best_model, X_train, y_train, best_n
    )

    # 4. Walk-forward backtesting
    predictions, probabilities = walk_forward_backtest(
        best_model, X_train, bullish_states, X_test, y_test
    )

    # 5. Performans değerlendirmesi
    evaluate(y_test, predictions, probabilities, test_df)


if __name__ == "__main__":
    main()