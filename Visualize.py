"""
BIST-100 HMM Projesi
Görselleştirme: 4 Grafik
========================
Yazar : Ural Şenyılmaz
Tarih : 2026

Çalıştır: py -3.12 visualize.py
Çıktı   : figures/ klasörüne 4 PNG dosyası
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import MinMaxScaler
import warnings
import os

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "DejaVu Sans"

# ── Ayarlar ───────────────────────────────────────────────────────────────────

RAW_PATH   = "data/bist100_raw.csv"
TRAIN_END  = "2021-12-31"
TEST_START = "2022-01-01"
N_STATES   = 3
N_INIT     = 30
FIGURES_DIR = "figures"

FEATURE_COLS = [
    "rsi", "macd", "bb_width", "volume_norm",
    "usdtry_chg", "brent_chg", "sp500_chg", "dax_chg",
]

# Rejim renkleri — tutarlı kullanım için burada tanımla
REGIME_COLORS = {
    "Boga" : "#2ecc71",   # yeşil
    "Yatay": "#f39c12",   # turuncu
    "Ayi"  : "#e74c3c",   # kırmızı
}

os.makedirs(FIGURES_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Veri ve Model (hmm_regime.py ile aynı)
# ══════════════════════════════════════════════════════════════════════════════

def load_and_train():
    """Veriyi yükler, modeli eğitir, rejim haritasını döndürür."""
    print("📂 Veri yükleniyor ve model eğitiliyor...")

    df = pd.read_csv(RAW_PATH, index_col=0, parse_dates=True)
    df = df[FEATURE_COLS + ["close", "target"]].dropna()

    train = df[df.index <= TRAIN_END]
    test  = df[df.index >= TEST_START]

    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(train[FEATURE_COLS].values)
    X_test  = scaler.transform(test[FEATURE_COLS].values)

    # Model eğitimi — hmm_regime.py ile birebir aynı
    best_model = None
    best_score = -np.inf
    for seed in range(N_INIT):
        model = GaussianHMM(
            n_components=N_STATES,
            covariance_type="diag",
            n_iter=1000, tol=1e-6,
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

    print(f"   ✓ Model eğitildi | log-likelihood: {best_score:.2f}")

    # Eğitim setinde rejim isimlerini belirle (ortalama getiriye göre)
    train_states = best_model.predict(X_train)
    train_returns = train["close"].pct_change().fillna(0).values

    state_mean = {}
    for s in range(N_STATES):
        mask = train_states == s
        state_mean[s] = train_returns[mask].mean() if mask.sum() > 0 else 0

    sorted_states = sorted(state_mean.items(), key=lambda x: x[1], reverse=True)
    regime_map = {}
    names = ["Boga", "Yatay", "Ayi"]
    for rank, (sid, _) in enumerate(sorted_states):
        regime_map[sid] = names[rank]

    print(f"   ✓ Rejim haritası: {regime_map}")

    # Walk-forward: test setinde rejimler ve pozisyonlar
    bull_states = {sid for sid, name in regime_map.items() if name == "Boga"}
    X_so_far = X_train.copy()
    test_regimes   = []
    test_positions = []

    for i in range(len(X_test)):
        hidden        = best_model.predict(X_so_far)
        current_state = hidden[-1]
        test_regimes.append(regime_map[current_state])
        test_positions.append(1 if current_state in bull_states else 0)
        X_so_far = np.vstack([X_so_far, X_test[i]])

    return best_model, X_train, train, test, train_states, \
           test_regimes, test_positions, regime_map


# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 1 — Geçiş Matrisi Isı Haritası
# ══════════════════════════════════════════════════════════════════════════════

def plot_transition_matrix(model, regime_map):
    """
    HMM geçiş matrisini (A) ısı haritası olarak çizer.
    Her hücre: o durumdan o duruma geçiş olasılığı.
    """
    print("🎨 Grafik 1: Geçiş Matrisi...")

    names   = ["Boga", "Yatay", "Ayi"]
    labels  = ["Boğa", "Yatay", "Ayı"]
    n       = N_STATES

    # Geçiş matrisini rejim sırasına göre yeniden düzenle
    # regime_map: state_id → isim; ters: isim → state_id
    name_to_id = {v: k for k, v in regime_map.items()}
    order      = [name_to_id[n_] for n_ in names]
    A_ordered  = model.transmat_[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    im = ax.imshow(A_ordered, cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Geçiş Olasılığı")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Sonraki Durum", fontsize=11)
    ax.set_ylabel("Mevcut Durum", fontsize=11)
    ax.set_title("HMM Durum Geçiş Matrisi", fontsize=13, fontweight="bold")

    for i in range(n):
        for j in range(n):
            val   = A_ordered[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=12, color=color, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig1_transition_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✓ Kaydedildi: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 2 — Kümülatif Getiri Karşılaştırması
# ══════════════════════════════════════════════════════════════════════════════

def plot_cumulative_returns(test, test_positions):
    """
    HMM strateji, buy-and-hold ve rastgele stratejinin
    kümülatif getirilerini tek grafikte gösterir.
    """
    print("🎨 Grafik 2: Kümülatif Getiri...")

    ret = test["close"].pct_change().fillna(0).values
    pos = np.array(test_positions)
    n   = min(len(ret), len(pos))
    ret = ret[:n]
    pos = pos[:n]

    np.random.seed(42)
    rand_pos = np.random.randint(0, 2, n)

    hmm_cum  = (1 + ret * pos).cumprod()
    bh_cum   = (1 + ret).cumprod()
    rand_cum = (1 + ret * rand_pos).cumprod()

    dates = test.index[:n]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(dates, bh_cum,   color="#2c3e50", linewidth=2,
            label=f"Al-Tut (Buy-and-Hold)  — %{(bh_cum[-1]-1)*100:.1f}")
    ax.plot(dates, hmm_cum,  color="#27ae60", linewidth=2.5,
            linestyle="-",
            label=f"HMM Rejim Stratejisi   — %{(hmm_cum[-1]-1)*100:.1f}")
    ax.plot(dates, rand_cum, color="#95a5a6", linewidth=1.5,
            linestyle="--",
            label=f"Rastgele Strateji      — %{(rand_cum[-1]-1)*100:.1f}")

    ax.axhline(1, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Tarih", fontsize=11)
    ax.set_ylabel("Kümülatif Getiri (Başlangıç = 1)", fontsize=11)
    ax.set_title("Strateji Karşılaştırması — Kümülatif Getiri (2022–2024)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig2_cumulative_returns.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✓ Kaydedildi: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 3 — Rejim Zaman Serisi
# ══════════════════════════════════════════════════════════════════════════════

def plot_regime_timeline(test, test_regimes):
    """
    Test döneminde BIST-100 fiyat seyrini ve altında
    her haftanın rejim rengini renkli şerit olarak gösterir.
    """
    print("🎨 Grafik 3: Rejim Zaman Serisi...")

    n       = min(len(test), len(test_regimes))
    dates   = test.index[:n]
    prices  = test["close"].values[:n]
    regimes = test_regimes[:n]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)

    # Üst panel: BIST-100 fiyat
    ax1.plot(dates, prices, color="#2c3e50", linewidth=1.5)
    ax1.set_ylabel("BIST-100 Endeks Değeri", fontsize=10)
    ax1.set_title("BIST-100 ve HMM Rejim Tespiti (2022–2024)",
                  fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Fiyat grafiğinin arka planına rejim rengi ekle
    regime_to_color = {
        "Boga" : "#d5f5e3",
        "Yatay": "#fef9e7",
        "Ayi"  : "#fadbd8",
    }
    for i in range(n - 1):
        ax1.axvspan(dates[i], dates[i+1],
                    alpha=0.4,
                    color=regime_to_color.get(regimes[i], "white"),
                    linewidth=0)

    # Alt panel: rejim şeridi
    for i in range(n - 1):
        color = REGIME_COLORS.get(regimes[i], "gray")
        ax2.axvspan(dates[i], dates[i+1],
                    alpha=0.85, color=color, linewidth=0)

    ax2.set_yticks([])
    ax2.set_ylabel("Rejim", fontsize=10)
    ax2.set_xlabel("Tarih", fontsize=10)

    # Legend
    patches = [
        mpatches.Patch(color=REGIME_COLORS["Boga"],  label="Boğa Rejimi"),
        mpatches.Patch(color=REGIME_COLORS["Yatay"], label="Yatay Rejim"),
        mpatches.Patch(color=REGIME_COLORS["Ayi"],   label="Ayı Rejimi"),
    ]
    ax2.legend(handles=patches, loc="lower right", fontsize=9,
               ncol=3, framealpha=0.9)

    fig.autofmt_xdate()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig3_regime_timeline.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✓ Kaydedildi: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# GRAFİK 4 — Eğitim Verisi Rejim Dağılımı
# ══════════════════════════════════════════════════════════════════════════════

def plot_regime_distribution(train, train_states, regime_map):
    """
    Eğitim verisinde (2010-2021) her rejimin kaç hafta sürdüğünü
    ve o dönemdeki ortalama haftalık getiriyi bar grafik ile gösterir.
    """
    print("🎨 Grafik 4: Rejim Dağılımı...")

    train_returns = train["close"].pct_change().fillna(0).values
    names_order   = ["Boga", "Yatay", "Ayi"]
    labels        = ["Boğa", "Yatay", "Ayı"]
    name_to_id    = {v: k for k, v in regime_map.items()}

    counts   = []
    means    = []
    colors   = []

    for name in names_order:
        sid  = name_to_id[name]
        mask = train_states == sid
        counts.append(mask.sum())
        means.append(train_returns[mask].mean() * 100)
        colors.append(REGIME_COLORS[name])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Sol: hafta sayısı
    bars1 = ax1.bar(labels, counts, color=colors, edgecolor="white",
                    linewidth=1.2, width=0.5)
    ax1.set_ylabel("Hafta Sayısı", fontsize=11)
    ax1.set_title("Eğitim Verisi Rejim Dağılımı\n(2010–2021)",
                  fontsize=11, fontweight="bold")
    ax1.set_ylim(0, max(counts) * 1.2)
    for bar, count in zip(bars1, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(count), ha="center", va="bottom", fontsize=11,
                 fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.3)

    # Sağ: ortalama haftalık getiri
    bars2 = ax2.bar(labels, means, color=colors, edgecolor="white",
                    linewidth=1.2, width=0.5)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Ort. Haftalık Getiri (%)", fontsize=11)
    ax2.set_title("Rejim Başına Ortalama Getiri\n(Eğitim Verisi)",
                  fontsize=11, fontweight="bold")
    for bar, mean in zip(bars2, means):
        offset = 0.03 if mean >= 0 else -0.08
        ax2.text(bar.get_x() + bar.get_width()/2,
                 mean + offset,
                 f"%{mean:+.2f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig4_regime_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✓ Kaydedildi: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# ANA FONKSİYON
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═"*55)
    print("  BIST-100 HMM — GÖRSELLEŞTİRME")
    print("═"*55 + "\n")

    # Veri yükle ve modeli eğit
    (model, X_train, train, test,
     train_states, test_regimes, test_positions,
     regime_map) = load_and_train()

    # 4 grafiği üret
    plot_transition_matrix(model, regime_map)
    plot_cumulative_returns(test, test_positions)
    plot_regime_timeline(test, test_regimes)
    plot_regime_distribution(train, train_states, regime_map)

    print("\n" + "═"*55)
    print(f"  ✅ 4 grafik figures/ klasörüne kaydedildi.")
    print("═"*55)


if __name__ == "__main__":
    main()