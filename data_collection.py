"""
BIST-100 Yön Tahmini Projesi
Adım 1: Veri Toplama ve Ön İşleme
========================================
Yazar : Ural Şenyılmaz
Tarih : 2026
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import requests
import warnings
import os

warnings.filterwarnings("ignore")

# ── Genel Ayarlar ──────────────────────────────────────────────────────────────

START_DATE = "2010-01-01"
END_DATE   = "2024-12-31"
OUTPUT_DIR = "data"          # CSV'ler buraya kaydedilir

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 1 — Fiyat Verisi (yfinance)
# ══════════════════════════════════════════════════════════════════════════════

def download_price_data() -> pd.DataFrame:
    """
    BIST-100 günlük OHLCV verisini çeker ve haftalık kapanışa dönüştürür.
    Raporda taahhüt: yfinance, 2010-2024, forward-fill, haftalık resample.
    """
    print("📥 BIST-100 fiyat verisi indiriliyor...")

    raw = yf.download("XU100.IS", start=START_DATE, end=END_DATE, progress=False)

    # Sütun adlarını düzleştir (MultiIndex gelebilir)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Günlük eksik değerleri forward-fill ile doldur (tatil günleri)
    raw = raw.ffill()

    # Haftalık resample — her haftanın son işlem günü kapanış fiyatı
    weekly = pd.DataFrame()
    weekly["close"]  = raw["Close"].resample("W").last()
    weekly["open"]   = raw["Open"].resample("W").first()
    weekly["high"]   = raw["High"].resample("W").max()
    weekly["low"]    = raw["Low"].resample("W").min()
    weekly["volume"] = raw["Volume"].resample("W").sum()

    # Hâlâ eksik satır varsa tekrar forward-fill
    weekly = weekly.ffill().dropna()

    print(f"   ✓ {len(weekly)} haftalık gözlem  |  {weekly.index[0].date()} → {weekly.index[-1].date()}")
    return weekly


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 2 — Teknik Göstergeler (pandas-ta)
# ══════════════════════════════════════════════════════════════════════════════

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rapora uygun teknik göstergeleri hesaplar:
      - RSI (14)
      - MACD (12-26-9) + sinyal çizgisi
      - SMA-20, SMA-50
      - EMA-20
      - Bollinger Bantları (20, 2 std)  → üst, alt, orta bant, bant genişliği
      - Normalize hacim (haftalık ortalamayla)
    """
    print("⚙️  Teknik göstergeler hesaplanıyor...")

    # pandas-ta için sütun adları küçük harf olmalı
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # RSI — 14 periyot
    df["rsi"] = ta.rsi(df["close"], length=14)

    # MACD — 12 hızlı, 26 yavaş, 9 sinyal
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"]        = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"]   = macd["MACDh_12_26_9"]   # histogram (fark)

    # Hareketli Ortalamalar
    df["sma_20"] = ta.sma(df["close"], length=20)
    df["sma_50"] = ta.sma(df["close"], length=50)
    df["ema_20"] = ta.ema(df["close"], length=20)

    # Bollinger Bantları — 20 periyot, 2 standart sapma
    bb = ta.bbands(df["close"], length=20, std=2)
    bb_upper_col = [c for c in bb.columns if c.startswith("BBU")][0]
    bb_mid_col   = [c for c in bb.columns if c.startswith("BBM")][0]
    bb_lower_col = [c for c in bb.columns if c.startswith("BBL")][0]
    bb_width_col = [c for c in bb.columns if c.startswith("BBB")][0]
    df["bb_upper"]  = bb[bb_upper_col]   # üst bant
    df["bb_mid"]    = bb[bb_mid_col]     # orta bant (= SMA-20)
    df["bb_lower"]  = bb[bb_lower_col]   # alt bant
    df["bb_width"]  = bb[bb_width_col]   # bant genişliği (volatilite göstergesi)

    # Normalize Hacim — her gözlemi 20 haftalık ortalama hacme böl
    df["volume_norm"] = df["volume"] / df["volume"].rolling(20).mean()

    print(f"   ✓ RSI, MACD, SMA-20/50, EMA-20, Bollinger, Hacim hesaplandı")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 3 — Makroekonomik Veriler
# ══════════════════════════════════════════════════════════════════════════════

def download_macro_data() -> pd.DataFrame:
    """
    yfinance üzerinden çekilebilen makroekonomik veriler:
      - USD/TRY kuru + haftalık değişim oranı
      - Brent ham petrol (BZ=F)
      - S&P 500 (^GSPC)
      - DAX (^GDAXI)

    TCMB faiz + TÜİK enflasyon:
      TCMB'nin açık API'si (evds.tcmb.gov.tr) üzerinden çekilir.
      API anahtarı gerekmektedir — aşağıda TCMB_API_KEY değişkenine gir.
      Anahtarı almak için: https://evds2.tcmb.gov.tr → Üye Ol → API Anahtarı
    """
    print("📥 Makroekonomik veriler indiriliyor...")

    tickers = {
        "usdtry" : "TRY=X",    # Dolar/TL
        "brent"  : "BZ=F",     # Ham petrol
        "sp500"  : "^GSPC",    # S&P 500
        "dax"    : "^GDAXI",   # DAX
    }

    macro = pd.DataFrame()

    for name, ticker in tickers.items():
        raw = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        series = raw["Close"].resample("W").last().ffill()
        macro[name] = series
        print(f"   ✓ {name} ({ticker}) indirildi")

    # Haftalık değişim oranları (pct_change)
    macro["usdtry_chg"] = macro["usdtry"].pct_change()
    macro["brent_chg"]  = macro["brent"].pct_change()
    macro["sp500_chg"]  = macro["sp500"].pct_change()
    macro["dax_chg"]    = macro["dax"].pct_change()

    # TCMB & TÜİK verileri
    macro = _add_tcmb_data(macro)

    return macro


def _add_tcmb_data(macro: pd.DataFrame) -> pd.DataFrame:
    """
    TCMB EVDS API'den politika faiz oranı ve TÜFE/ÜFE verilerini çeker.
    API anahtarın yoksa mock veri ile devam eder (geliştirme aşaması için).
    """

    TCMB_API_KEY = "BURAYA_API_ANAHTARINI_GIR"  # ← evds2.tcmb.gov.tr'den al

    # Seri kodları
    # TP.TG2.Y01 = TCMB politika faiz oranı (gecelik)
    # TP.FG.J0   = TÜFE (aylık, bir önceki aya göre %)
    # TP.FG.J14  = ÜFE (aylık, bir önceki aya göre %)
    series_map = {
        "tcmb_faiz" : "TP.TG2.Y01",
        "tufe"      : "TP.FG.J0",
        "ufe"       : "TP.FG.J14",
    }

    if TCMB_API_KEY == "BURAYA_API_ANAHTARINI_GIR":
        print("   ⚠️  TCMB API anahtarı girilmedi — şimdilik NaN bırakılıyor")
        print("      → evds2.tcmb.gov.tr'den ücretsiz anahtar alıp koda ekle")
        macro["tcmb_faiz"] = np.nan
        macro["tufe"]      = np.nan
        macro["ufe"]       = np.nan
        return macro

    base_url = "https://evds2.tcmb.gov.tr/service/evds"

    for col_name, series_code in series_map.items():
        try:
            url = (
                f"{base_url}/series={series_code}"
                f"&startDate={START_DATE}&endDate={END_DATE}"
                f"&type=json&key={TCMB_API_KEY}&frequency=5"  # 5 = aylık
            )
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            records = data["items"]
            dates  = [r["Tarih"] for r in records]
            values = [r.get(series_code.replace(".", "_"), np.nan) for r in records]

            s = pd.Series(
                [float(v) if v not in (None, "", " ") else np.nan for v in values],
                index=pd.to_datetime(dates, format="%d-%m-%Y"),
                name=col_name
            )

            # Aylık → haftalık: forward-fill (ay boyunca aynı değer geçerli)
            s = s.resample("W").last().ffill()
            macro[col_name] = s
            print(f"   ✓ {col_name} TCMB'den indirildi")

        except Exception as e:
            print(f"   ✗ {col_name} indirilemedi: {e}")
            macro[col_name] = np.nan

    return macro


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 4 — Birleştirme, Hedef Etiket, Normalizasyon
# ══════════════════════════════════════════════════════════════════════════════

def merge_and_label(price_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fiyat + teknik gösterge verisini makroekonomik veriyle birleştirir.
    Hedef etiket (0/1) ve normalizasyon uygulanır.
    """
    print("🔗 Veriler birleştiriliyor...")

    df = price_df.join(macro_df, how="left")

    # ── Hedef Değişken ────────────────────────────────────────────────────────
    # 1 = bu haftanın kapanışı > geçen haftanın kapanışı (artış)
    # 0 = azalış veya değişim yok
    df["target"] = (df["close"] > df["close"].shift(1)).astype(int)

    # Sınıf dağılımını kontrol et
    dist = df["target"].value_counts(normalize=True) * 100
    print(f"   ✓ Hedef etiket oluşturuldu  |  Artış: %{dist.get(1,0):.1f}  Azalış: %{dist.get(0,0):.1f}")

    # ── Kalan Eksik Değerler ──────────────────────────────────────────────────
    # İlk birkaç satırda göstergeler NaN olur (warmup periyodu), onları at
    before = len(df)
    df = df.dropna(subset=["rsi", "macd", "sma_50"])
    print(f"   ✓ Warmup satırları temizlendi  |  {before} → {len(df)} gözlem")

    # Kalan NaN'ları forward-fill (özellikle makro veriler)
    df = df.ffill()

    # ── Normalizasyon (Min-Max) ───────────────────────────────────────────────
    # ÖNEMLI: Sadece eğitim setine fit edilecek — burada henüz tüm veriye
    # uyguluyoruz; train/test split sonrası scaler tekrar fit edilecek.
    # Bu adım veriyi [0,1] aralığına çeker.
    exclude_cols = ["target"]   # etiket normalize edilmez
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    df_norm = df.copy()
    for col in feature_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max != col_min:
            df_norm[col] = (df[col] - col_min) / (col_max - col_min)
        # min == max ise sütun sabit → 0 bırak (nadiren olur)

    print(f"   ✓ Min-Max normalizasyonu uygulandı  |  Toplam özellik: {len(feature_cols)}")
    return df_norm, df   # normalize + ham veri ikisini de döndür


# ══════════════════════════════════════════════════════════════════════════════
# ADIM 5 — Kaydet ve Özet
# ══════════════════════════════════════════════════════════════════════════════

def save_and_summarize(df_norm: pd.DataFrame, df_raw: pd.DataFrame):
    """
    İşlenmiş veriyi CSV olarak kaydeder ve özet bilgi basar.
    """
    norm_path = os.path.join(OUTPUT_DIR, "bist100_normalized.csv")
    raw_path  = os.path.join(OUTPUT_DIR, "bist100_raw.csv")

    df_norm.to_csv(norm_path)
    df_raw.to_csv(raw_path)

    print("\n" + "═"*60)
    print("📊 VERİ SETİ ÖZETİ")
    print("═"*60)
    print(f"  Dönem          : {df_raw.index[0].date()} → {df_raw.index[-1].date()}")
    print(f"  Haftalık gözlem: {len(df_raw)}")
    print(f"  Toplam özellik : {len(df_raw.columns) - 1}  (+1 hedef etiket)")
    print(f"  Artış oranı    : %{df_raw['target'].mean()*100:.1f}")
    print(f"  Azalış oranı   : %{(1-df_raw['target'].mean())*100:.1f}")
    print()
    print("  Özellikler:")
    for col in df_raw.columns:
        eksik = df_raw[col].isna().sum()
        print(f"    {col:<20} | eksik: {eksik}")
    print()
    print(f"  ✅ Normalize veri  → {norm_path}")
    print(f"  ✅ Ham veri        → {raw_path}")
    print("═"*60)


# ══════════════════════════════════════════════════════════════════════════════
# ANA FONKSİYON
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═"*60)
    print("  BIST-100 YÖN TAHMİNİ — VERİ TOPLAMA AŞAMASI")
    print("═"*60 + "\n")

    # 1. Ham fiyat verisi
    price_df = download_price_data()

    # 2. Teknik göstergeler ekle
    price_df = add_technical_indicators(price_df)

    # 3. Makroekonomik veriler
    macro_df = download_macro_data()

    # 4. Birleştir, etiketle, normalize et
    df_norm, df_raw = merge_and_label(price_df, macro_df)

    # 5. Kaydet ve özet bas
    save_and_summarize(df_norm, df_raw)


if __name__ == "__main__":
    main()
