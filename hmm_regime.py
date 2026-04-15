"""
BIST-100 Yön Tahmini Projesi
Adım 3: HMM Rejim Tespiti ve Al-Sat Stratejisi
================================================
Yazar : Ural Şenyılmaz
Tarih : 2026

Bu dosyada HMM, fiyat yönü tahmini için değil literatürdeki
asıl kullanım amacına uygun olarak piyasa rejimi tespiti için
kullanılmaktadır. Model üç gizli durum öğrenir:
  - Boğa Rejimi  : düşük volatilite, sürekli yükseliş
  - Ayı Rejimi   : yüksek volatilite, düşüş baskısı
  - Yatay Rejim  : belirsiz yön, orta volatilite

Strateji:
  - Boğa rejiminde → pozisyon tut (al)
  - Ayı / Yatay rejiminde → pozisyondan çık (sat)

Başarı kriteri:
  - Buy-and-hold getirisinin en az %50'sine ulaşmak
  - Maksimum drawdown'ı buy-and-hold'dan düşük tutmak
  - Sharpe oranını buy-and-hold'dan yüksek tutmak
"""

import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import MinMaxScaler
import warnings
import os

warnings.filterwarnings("ignore")

# ── Ayarlar ────────────────────────────────────────────────────────────────────

RAW_PATH   = "data/bist100_raw.csv"
OUTPUT_DIR = "data"

TRAIN_END  = "2021-12-31"
TEST_START = "2022-01-01"

N_STATES = 3    # Boğa / Ayı / Yatay — raporda taahhüt edilen
N_INIT   = 30   # Daha fazla başlangıç noktası → daha stabil sonuç

# Rejim tespiti için özellik seti
# Volatilite + trend + makro — rejim ayrışması için en bilgilendirici set
FEATURE_COLS = [
    "rsi",
    "macd",
    "bb_width",       # volatilite — rejim tespitinde kritik
    "volume_norm",
    "usdtry_chg",
    "brent_chg",
    "sp500_chg",
    "dax_chg",
]


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 1 — Veri Yükle
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    """
    Ham veriyi yükler, eğitim/test olarak böler.
    Scaler yalnızca eğitim setine fit edilir.
    """
    print("📂 Veri yükleniyor...")

    df = pd.read_csv(RAW_PATH, index_col=0, parse_dates=True)
    df = df[FEATURE_COLS + ["close", "target"]].dropna()

    train = df[df.index <= TRAIN_END]
    test  = df[df.index >= TEST_START]

    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(train[FEATURE_COLS].values)
    X_test  = scaler.transform(test[FEATURE_COLS].values)

    print(f"   ✓ Eğitim : {len(train)} hafta  ({train.index[0].date()} → {train.index[-1].date()})")
    print(f"   ✓ Test   : {len(test)}  hafta  ({test.index[0].date()} → {test.index[-1].date()})")

    return X_train, X_test, train, test, scaler


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 2 — HMM Eğitimi (N=3, Baum-Welch)
# ══════════════════════════════════════════════════════════════════════════════

def train_hmm(X_train):
    """
    N=3 GaussianHMM eğitir.
    N_INIT farklı başlangıç noktasından en yüksek log-likelihood'u seçer.
    Baum-Welch (EM) algoritması hmmlearn tarafından otomatik uygulanır.
    """
    print(f"\n🔧 HMM eğitiliyor (N={N_STATES}, {N_INIT} başlangıç noktası)...")

    best_model = None
    best_score = -np.inf

    for seed in range(N_INIT):
        model = GaussianHMM(
            n_components=N_STATES,
            covariance_type="diag",
            n_iter=1000,
            tol=1e-6,
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

    print(f"   ✓ En iyi log-likelihood: {best_score:.2f}")
    return best_model


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 3 — Rejim Tespiti ve İsimlendirme (Viterbi)
# ══════════════════════════════════════════════════════════════════════════════

def detect_regimes(model, X_train, train_df):
    """
    Viterbi algoritmasıyla eğitim verisindeki gizli durum dizisini çıkarır.
    Her durumu karakterize etmek için:
      - Ortalama haftalık getiri
      - Getiri standart sapması (volatilite)
      - Artış oranı
    hesaplanır ve durumlar Boğa / Yatay / Ayı olarak isimlendirilir.
    """
    print("\n📊 Rejimler tespit ediliyor (Viterbi — eğitim verisi)...")

    hidden_states = model.predict(X_train)
    returns       = train_df["close"].pct_change().fillna(0).values

    state_stats = {}
    for s in range(N_STATES):
        mask = hidden_states == s
        if mask.sum() == 0:
            continue
        state_returns = returns[mask]
        state_stats[s] = {
            "mean_return" : state_returns.mean(),
            "volatility"  : state_returns.std(),
            "up_rate"     : (state_returns > 0).mean(),
            "count"       : int(mask.sum()),
        }

    # Ortalama getiriye göre sırala → en yüksek = Boğa, en düşük = Ayı
    sorted_states = sorted(state_stats.items(),
                           key=lambda x: x[1]["mean_return"], reverse=True)

    regime_map   = {}   # state_id → rejim adı
    regime_names = ["🐂 Boğa", "↔️  Yatay", "🐻 Ayı"]

    print(f"\n   {'Durum':<8} {'Rejim':<12} {'Ort.Getiri':>10} {'Volatilite':>11} {'Artış':>7} {'Hafta':>6}")
    print(f"   {'-'*58}")

    for rank, (state_id, stats) in enumerate(sorted_states):
        name = regime_names[rank]
        regime_map[state_id] = name
        print(f"   Durum {state_id}  {name:<12} "
              f"%{stats['mean_return']*100:>+7.2f}    "
              f"%{stats['volatility']*100:>8.2f}    "
              f"%{stats['up_rate']*100:>5.1f}   "
              f"{stats['count']:>5}")

    # Boğa durumlarını belirle (al sinyali verecekler)
    bull_states = {state_id for state_id, name in regime_map.items()
                   if "Boğa" in name}

    return hidden_states, regime_map, bull_states, state_stats


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 4 — Walk-Forward Backtesting
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward_backtest(model, X_train, bull_states, X_test, test_df):
    """
    Walk-forward backtesting — her adımda model yalnızca geçmişi görür.

    Strateji:
      - Viterbi ile mevcut rejimi tespit et
      - Boğa rejimindeyse → o hafta pozisyonda kal (getiriyi al)
      - Ayı/Yatay rejimindeyse → pozisyondan çık (o haftanın getirisi 0)

    Komisyon ve vergi ihmal edilmektedir (raporda belirtilmiştir).
    """
    print("\n⏱️  Walk-forward backtesting çalışıyor...")

    regimes      = []
    positions    = []   # 1 = pozisyonda, 0 = nakit
    X_so_far     = X_train.copy()

    weekly_returns = test_df["close"].pct_change().fillna(0).values

    for i in range(len(X_test)):
        hidden        = model.predict(X_so_far)
        current_state = hidden[-1]

        position = 1 if current_state in bull_states else 0
        positions.append(position)
        regimes.append(current_state)

        X_so_far = np.vstack([X_so_far, X_test[i]])

    positions = np.array(positions)
    regimes   = np.array(regimes)

    # Pozisyonda olunan hafta sayısı
    bull_weeks = positions.sum()
    bear_weeks = (positions == 0).sum()
    print(f"   ✓ {len(positions)} haftalık tahmin tamamlandı")
    print(f"   Pozisyonda (Boğa) : {bull_weeks} hafta (%{bull_weeks/len(positions)*100:.1f})")
    print(f"   Nakitte   (Ayı)   : {bear_weeks} hafta (%{bear_weeks/len(positions)*100:.1f})")

    return positions, regimes, weekly_returns


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 5 — Performans Değerlendirmesi
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(positions, weekly_returns, test_df, regime_map, regimes):
    """
    Strateji performansını buy-and-hold ile karşılaştırır.

    Metrikler:
      - Toplam getiri
      - Maksimum drawdown (en derin tepe-dip düşüşü)
      - Sharpe oranı (getiri / volatilite)
      - Pozisyonda geçirilen süre
    """
    min_len = min(len(positions), len(weekly_returns))
    pos     = positions[:min_len]
    ret     = weekly_returns[:min_len]

    # ── Strateji Getirisi ─────────────────────────────────────────────────────
    strategy_weekly  = pd.Series(ret * pos)
    strategy_cumret  = (1 + strategy_weekly).cumprod()
    strategy_total   = (strategy_cumret.iloc[-1] - 1) * 100

    # ── Buy-and-Hold Getirisi ─────────────────────────────────────────────────
    bh_cumret  = (1 + pd.Series(ret)).cumprod()
    bh_total   = (bh_cumret.iloc[-1] - 1) * 100

    # ── Maksimum Drawdown ─────────────────────────────────────────────────────
    def max_drawdown(cumret_series):
        peak = cumret_series.cummax()
        dd   = (cumret_series - peak) / peak
        return dd.min() * 100

    strategy_mdd = max_drawdown(strategy_cumret)
    bh_mdd       = max_drawdown(bh_cumret)

    # ── Sharpe Oranı (yıllık, risksiz faiz = 0 varsayımı) ────────────────────
    def sharpe(weekly_ret_array):
        r = pd.Series(weekly_ret_array)
        if r.std() == 0:
            return 0
        return (r.mean() / r.std()) * np.sqrt(52)   # 52 hafta/yıl

    strategy_sharpe = sharpe(strategy_weekly)
    bh_sharpe       = sharpe(ret)

    # ── Sonuçları Yazdır ─────────────────────────────────────────────────────
    print("\n" + "═"*62)
    print("📈 STRATEJİ PERFORMANSI (Test: 2022-2024)")
    print("═"*62)
    print(f"  {'Metrik':<28} {'HMM Strateji':>12} {'Buy-and-Hold':>12}")
    print(f"  {'-'*54}")
    print(f"  {'Toplam Getiri':<28} %{strategy_total:>10.1f}   %{bh_total:>10.1f}")
    print(f"  {'Maks. Drawdown':<28} %{strategy_mdd:>10.1f}   %{bh_mdd:>10.1f}")
    print(f"  {'Sharpe Oranı':<28} {strategy_sharpe:>11.2f}   {bh_sharpe:>11.2f}")
    print(f"  {'Pozisyonda Geçirilen Süre':<28} %{pos.mean()*100:>10.1f}   %{'100.0':>10}")
    print(f"  {'-'*54}")

    # Başarı değerlendirmesi
    bh_50pct = bh_total * 0.5
    print(f"\n  Buy-and-Hold getirisi        : %{bh_total:.1f}")
    print(f"  Hedef (BH'nin %%50'si)        : %{bh_50pct:.1f}")
    print(f"  HMM Strateji getirisi        : %{strategy_total:.1f}")

    if strategy_total >= bh_total:
        print(f"\n  ✅ Buy-and-Hold AŞILDI!")
    elif strategy_total >= bh_50pct:
        print(f"\n  ✅ Hedef karşılandı (BH'nin %%50'si aşıldı)")
    else:
        print(f"\n  ⚠️  Hedef karşılanamadı")

    if strategy_mdd > bh_mdd:
        print(f"  ✅ Drawdown iyileştirildi (daha az risk)")

    if strategy_sharpe > bh_sharpe:
        print(f"  ✅ Sharpe oranı iyileştirildi (risk-ayarlı getiri daha iyi)")

    # ── Baseline 1: Rastgele Strateji ────────────────────────────────────────
    np.random.seed(42)
    random_pos          = np.random.randint(0, 2, min_len)
    random_weekly       = pd.Series(ret * random_pos)
    random_cumret       = (1 + random_weekly).cumprod()
    random_total        = (random_cumret.iloc[-1] - 1) * 100
    random_mdd          = max_drawdown(random_cumret)
    random_sharpe       = sharpe(random_weekly.values)

    # ── Baseline 2: Her Zaman Pozisyonda (Naive Al-Tut) ──────────────────────
    # Buy-and-hold ile aynı — zaten yukarıda hesaplandı

    # ── Baseline 3: Her Zaman Nakitte ────────────────────────────────────────
    always_cash_total  = 0.0
    always_cash_mdd    = 0.0
    always_cash_sharpe = 0.0

    print()
    print("  Baseline Karşılaştırması:")
    print(f"  {'Strateji':<28} {'Getiri':>8} {'Drawdown':>10} {'Sharpe':>8}")
    print(f"  {'-'*56}")
    print(f"  {'HMM Rejim Stratejisi':<28} %{strategy_total:>6.1f}   %{strategy_mdd:>7.1f}   {strategy_sharpe:>6.2f}")
    print(f"  {'Buy-and-Hold':<28} %{bh_total:>6.1f}   %{bh_mdd:>7.1f}   {bh_sharpe:>6.2f}")
    print(f"  {'Rastgele Strateji':<28} %{random_total:>6.1f}   %{random_mdd:>7.1f}   {random_sharpe:>6.2f}")
    print(f"  {'Her Zaman Nakitte':<28} %{always_cash_total:>6.1f}   %{always_cash_mdd:>7.1f}   {always_cash_sharpe:>6.2f}")
    print(f"  {'-'*56}")

    print("═"*62)

    # ── Sonuçları Kaydet ─────────────────────────────────────────────────────
    result_df = pd.DataFrame({
        "tarih"           : test_df.index[:min_len],
        "haftalik_getiri" : ret,
        "pozisyon"        : pos,
        "strateji_getiri" : strategy_weekly,
        "strateji_kumulatif" : strategy_cumret.values,
        "bh_kumulatif"    : bh_cumret.values,
        "rejim"           : [regime_map.get(r, "?") for r in regimes[:min_len]],
    })
    out_path = os.path.join(OUTPUT_DIR, "regime_backtest_results.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\n  ✅ Sonuçlar → {out_path}")

    return strategy_total, bh_total, strategy_mdd, bh_mdd


# ══════════════════════════════════════════════════════════════════════════════
# ANA FONKSİYON
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═"*62)
    print("  BIST-100 — HMM REJİM TESPİTİ VE AL-SAT STRATEJİSİ")
    print("═"*62)

    # 1. Veri yükle
    X_train, X_test, train_df, test_df, scaler = load_data()

    # 2. HMM eğit
    model = train_hmm(X_train)

    # 3. Rejimleri tespit et ve isimlendir
    hidden_states, regime_map, bull_states, state_stats = detect_regimes(
        model, X_train, train_df
    )

    # 4. Walk-forward backtesting
    positions, regimes, weekly_returns = walk_forward_backtest(
        model, X_train, bull_states, X_test, test_df
    )

    # 5. Performans değerlendirmesi
    evaluate(positions, weekly_returns, test_df, regime_map, regimes)


if __name__ == "__main__":
    main()