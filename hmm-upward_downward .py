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

TRAIN_END  = "2021-12-31"
TEST_START = "2022-01-01"

N_STATES_LIST = [2, 3, 4]
N_INIT        = 20

FEATURE_COLS = [
    "rsi", "macd", "bb_width", "volume_norm",
    "usdtry_chg", "brent_chg", "sp500_chg", "dax_chg",
]


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 1 — Veri Yükle ve Böl
# ══════════════════════════════════════════════════════════════════════════════

def load_and_split():
    """
    Ham veriyi yükler, eğitim/test olarak böler.
    MinMaxScaler yalnızca eğitim setine fit edilir — data leakage önlenir.
    """
    print("📂 Veri yükleniyor...")

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    cols = FEATURE_COLS + ["target"]
    df   = df[cols].dropna()

    train = df[df.index <= TRAIN_END]
    test  = df[df.index >= TEST_START]

    X_train = train[FEATURE_COLS].values
    y_train = train["target"].values
    X_test  = test[FEATURE_COLS].values
    y_test  = test["target"].values

    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"   ✓ Eğitim : {len(train)} hafta  ({train.index[0].date()} → {train.index[-1].date()})")
    print(f"   ✓ Test   : {len(test)}  hafta  ({test.index[0].date()} → {test.index[-1].date()})")
    print(f"   ✓ Özellik sayısı: {len(FEATURE_COLS)}")

    return X_train, y_train, X_test, y_test, train, test


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 2 — HMM Eğitimi
# ══════════════════════════════════════════════════════════════════════════════

def train_hmm(X_train, n_states, n_init=N_INIT):
    """
    GaussianHMM eğitir. Baum-Welch (EM) hmmlearn tarafından otomatik kullanılır.
    n_init farklı random_state ile en yüksek log-likelihood'u seçer.
    """
    best_model = None
    best_score = -np.inf

    for seed in range(n_init):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=500,
            tol=1e-4,
            random_state=seed,
        )
        try:
            model.fit(X_train)
            score = model.score(X_train)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    return best_model, best_score


def select_best_n(X_train, n_states_list=N_STATES_LIST):
    """
    N=2,3,4 için BIC/AIC hesaplar, en düşük BIC'li modeli seçer.
    diag covariance parametre sayısı: n*(n-1) + n*d + n*d
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

        d        = n_features
        k_params = (n * (n - 1)) + (n * d) + (n * d)
        bic      = -2 * log_likelihood + k_params * np.log(n_samples)
        aic      = -2 * log_likelihood + 2 * k_params

        results.append({
            "n_states"       : n,
            "log_likelihood" : round(log_likelihood, 2),
            "bic"            : round(bic, 2),
            "aic"            : round(aic, 2),
            "model"          : model,
        })

        print(f"   N={n} | LogL: {log_likelihood:>10.2f} | BIC: {bic:>12.2f} | AIC: {aic:>12.2f}")

    best = min(results, key=lambda x: x["bic"])
    print(f"\n   ✅ Seçilen model: N={best['n_states']} (en düşük BIC)")

    return best["model"], best["n_states"], results


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 3 — Durum → Artış Oranı Eşlemesi (Viterbi)
# ══════════════════════════════════════════════════════════════════════════════

def learn_state_up_rates(model, X_train, y_train, n_states):
    """
    Viterbi ile eğitim verisindeki gizli durum dizisini çıkarır.
    Her durum için gerçek artış oranını hesaplar.

    Eşik (threshold): tüm durumların artış oranı ortalaması.
    Bir durumun artış oranı eşiğin üzerindeyse → bullish (tahmin=1)
    Altındaysa → bearish (tahmin=0).
    Bu sayede mutlak %50 yerine göreceli karşılaştırma yapılır;
    piyasanın genel eğiliminden daha iyi durumlara 1, kötü olanlara 0 denir.
    """
    hidden_states = model.predict(X_train)   # Viterbi

    state_up_rate = {}
    for s in range(n_states):
        mask    = hidden_states == s
        count   = mask.sum()
        up_rate = y_train[mask].mean() if count > 0 else 0.5
        state_up_rate[s] = up_rate

    # Eşik: durumların ağırlıklı ortalaması (hafta sayısına göre)
    total = len(hidden_states)
    threshold = sum(
        rate * (hidden_states == s).sum() / total
        for s, rate in state_up_rate.items()
    )

    # Artış oranına göre sırala, rejim ismi ver
    sorted_states = sorted(state_up_rate.items(), key=lambda x: x[1])
    if n_states == 2:
        regime_names = ["Ayı (Düşüş)", "Boğa (Yükseliş)"]
    elif n_states == 3:
        regime_names = ["Ayı (Düşüş)", "Yatay", "Boğa (Yükseliş)"]
    else:
        regime_names = [f"Rejim {i+1}" for i in range(n_states)]

    print(f"\n📊 Gizli Durum Yorumu (Viterbi — eğitim verisi):")
    print(f"   Dinamik eşik (threshold): %{threshold*100:.1f}")
    for rank, (s, rate) in enumerate(sorted_states):
        regime  = regime_names[rank] if rank < len(regime_names) else f"Durum {s}"
        count   = int((hidden_states == s).sum())
        sinyal  = "↑ Artış" if rate >= threshold else "↓ Düşüş"
        print(f"   Durum {s}: {regime:<22} | "
              f"Artış oranı: %{rate*100:.1f} | "
              f"Hafta: {count:>3} | Sinyal: {sinyal}")

    return state_up_rate, threshold


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 4 — Walk-Forward Backtesting
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward_backtest(model, X_train, state_up_rate, threshold, X_test, y_test):
    """
    Walk-forward backtesting.
    Tahmin mekanizması:
      1. Viterbi ile mevcut gizli durumu bul
      2. O durumun eğitimde ölçülen artış oranını al
      3. Artış oranı >= threshold → tahmin=1, değilse → tahmin=0
    Threshold sabit %50 değil; eğitim verisinin genel artış oranı
    kullanılır. Böylece tüm durumlar %50 üzerinde olsa bile
    görece zayıf olanlar düşüş sinyali verir.
    """
    print("\n⏱️  Walk-forward backtesting çalışıyor...")

    predictions   = []
    probabilities = []
    X_so_far      = X_train.copy()

    for i in range(len(X_test)):
        hidden        = model.predict(X_so_far)   # Viterbi
        current_state = hidden[-1]

        up_prob = state_up_rate.get(current_state, threshold)
        pred    = 1 if up_prob >= threshold else 0
        predictions.append(pred)
        probabilities.append(round(up_prob, 4))

        X_so_far = np.vstack([X_so_far, X_test[i]])

    predictions   = np.array(predictions)
    probabilities = np.array(probabilities)

    print(f"   ✓ {len(predictions)} haftalık tahmin tamamlandı")
    print(f"   Artış tahmini : {predictions.sum()} hafta")
    print(f"   Düşüş tahmini : {(predictions==0).sum()} hafta")

    return predictions, probabilities


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 5 — Performans Değerlendirmesi
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(y_test, predictions, probabilities, test_df):
    """
    Accuracy, Precision, Recall, F1, Confusion Matrix + baseline karşılaştırması.
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

    # Baseline 1: rastgele
    np.random.seed(42)
    random_acc = accuracy_score(y_test, np.random.randint(0, 2, len(y_test)))

    # Baseline 2: her zaman artış
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

    # Baseline 3: buy-and-hold vs strateji
    raw      = pd.read_csv(RAW_PATH, index_col=0, parse_dates=True)
    raw_test = raw[raw.index >= TEST_START]["close"]

    if len(raw_test) > 1:
        bh_return       = (raw_test.iloc[-1] / raw_test.iloc[0] - 1) * 100
        weekly_returns  = raw_test.pct_change().dropna().values
        min_len         = min(len(predictions), len(weekly_returns))
        strategy_return = (weekly_returns[:min_len] * predictions[:min_len]).sum() * 100

        print()
        print("  Piyasa Simülasyonu (test dönemi):")
        print(f"  Buy-and-Hold getirisi : %{bh_return:.1f}")
        print(f"  HMM strateji getirisi : %{strategy_return:.1f}")
        print("  (Komisyon/vergi ihmal edilmiştir)")

    print("═"*60)

    result_df = pd.DataFrame({
        "tarih"           : test_df.index[:len(predictions)],
        "gercek"          : y_test[:len(predictions)],
        "tahmin"          : predictions,
        "artis_olasiligi" : probabilities,
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

    X_train, y_train, X_test, y_test, train_df, test_df = load_and_split()

    best_model, best_n, all_results = select_best_n(X_train)

    state_up_rate, threshold = learn_state_up_rates(best_model, X_train, y_train, best_n)

    predictions, probabilities = walk_forward_backtest(
        best_model, X_train, state_up_rate, threshold, X_test, y_test
    )

    evaluate(y_test, predictions, probabilities, test_df)


if __name__ == "__main__":
    main()