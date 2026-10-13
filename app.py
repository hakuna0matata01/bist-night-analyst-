"""
BIST Akşam Taraması - Faz 1 (v2 - Güçlendirilmiş Motor)
--------------------------------------------------------
Piyasa kapandıktan sonra (örn. saat 20:00) çalıştırılarak, ertesi gün yükseliş
potansiyeli yüksek görünen BIST hisselerini teknik kriterlere göre puanlayan
ve sıralayan bir tarama aracı.

PUANLAMA (v2) — maksimum 7 pozitif puan, 2 risk cezası:

  Pozitif kriterler (her biri +1):
    1. Trend     : Fiyat > EMA20 > EMA50 (boğa dizilişi, sadece fiyat>EMA değil)
    2. Momentum  : RSI(14) 40-65 bandında ve yükseliyor
    3. MACD      : MACD çizgisi sinyalin üzerinde ve histogram güçleniyor
    4. Hacim     : Son günün hacmi 20g ortalamanın 1.3x üzerinde + fiyat yukarı
    5. Seviye    : Direnç kırılımı / direnç yakınlığı ya da sıkışma sonrası
    6. ADX       : ADX(14) ≥ eşik (trend gerçekten güçlü mü, yoksa yatay mı?)
    7. Göreli Güç: Son 20 günde BIST100 endeksinden daha güçlü performans

  Risk cezaları (her biri -1, toplam puan 0'ın altına inmez):
    A. Aşırı uzama : Fiyat, EMA20'nin %10+ üzerinde (kovalama/geri çekilme riski)
    B. Marj sınırı : Bugünkü değişim zaten ±%9+ (yarın için gap/tepki riski)

  Ek bağlam:
    - BIST100 endeksinin kendi trendi (EMA50 üzerinde mi) üstte bir uyarı/bilgi
      panosu olarak gösterilir.
    - Likidite filtresi: 20 günlük ortalama TL cirosu belirlenen eşiğin
      altındaki hisseler izleme listesinden (isteğe bağlı) hariç tutulur.
    - Tarihsel Başarı Oranı: Her hissenin KENDİ geçmiş verisinde, aynı puan
      eşiğini geçtiği günlerden sonraki günün pozitif kapanma oranı ve
      ortalama getirisi hesaplanır. Bu, kuralların o hissede gerçekten işe
      yarayıp yaramadığına dair ampirik bir gösterge sağlar (yeterli sinyal
      yoksa "veri yetersiz" gösterilir).

NOT: Bu araç yatırım tavsiyesi değildir, sadece bir tarama/ön-eleme aracıdır.
Tarihsel başarı oranı geçmiş performanstır, gelecekteki sonucu garanti etmez.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ----------------------------------------------------------------------------
# Sayfa ayarları ve sabitler
# ----------------------------------------------------------------------------
st.set_page_config(page_title="BIST Akşam Taraması - Faz 1", layout="wide", page_icon="📈")

SAMPLE_TICKERS = [
    "AKBNK", "ALARK", "ASELS", "ASTOR", "BIMAS", "EKGYO", "ENKAI", "EREGL",
    "FROTO", "GARAN", "GUBRF", "HALKB", "ISCTR", "KCHOL", "KONTR", "KOZAL",
    "KRDMD", "ODAS", "OYAKC", "PETKM", "PGSUS", "SAHOL", "SASA", "SISE",
    "TAVHL", "TCELL", "THYAO", "TOASO", "TUPRS", "VAKBN", "YKBNK",
]

TICKERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bist_tickers.txt")
INDEX_TICKER = "XU100.IS"  # BIST 100 endeksi (piyasa rejimi / göreli güç referansı)

MIN_BARS_REQUIRED = 80      # EMA50, ADX ve sıkışma analizi için gereken minimum gün sayısı
DOWNLOAD_CHUNK_SIZE = 100   # Toplu indirmede tek seferde istenecek hisse sayısı
MAX_POSITIVE_CRITERIA = 7   # Pozitif kriter sayısı (puanın üst sınırı)
OVEREXTENSION_PCT = 0.10    # EMA20'nin bu oranın üzerinde ise "aşırı uzamış" say
NEAR_LIMIT_PCT = 9.0        # Günlük değişim bu yüzdeyi geçerse "marj sınırına yakın" say
MIN_BACKTEST_SIGNALS = 5    # Tarihsel başarı oranını göstermek için gereken min. sinyal sayısı
DEFAULT_TRIGGER_BUFFER_PCT = 0.5  # Giriş tetik seviyesi için referans direncin üzerine eklenen tampon (%)
SUSPICIOUS_JUMP_PCT = 30.0  # Tek günlük |değişim| bu eşiği geçerse "olası ayarlanmamış kurumsal işlem" say
MIN_AGGREGATE_BACKTEST_SIGNALS = 30  # Tüm-evren (cross-sectional) backtest için gereken min. sinyal-gün sayısı
RSI_BREAKOUT_UPPER = 80     # Kırılım eşliğinde momentum için izin verilen üst RSI sınırı (aşırı alım sınırı)


@st.cache_data(show_spinner=False)
def load_full_bist_list() -> list:
    """data/bist_tickers.txt içindeki tüm BIST hisse kodlarını okur."""
    try:
        with open(TICKERS_FILE, "r", encoding="utf-8") as f:
            codes = [line.strip().upper() for line in f if line.strip()]
        return sorted(set(codes))
    except FileNotFoundError:
        return SAMPLE_TICKERS


def normalize_ticker(raw: str) -> str:
    raw = raw.strip().upper()
    if not raw:
        return raw
    if not raw.endswith(".IS"):
        raw += ".IS"
    return raw


# ----------------------------------------------------------------------------
# Veri çekme
# ----------------------------------------------------------------------------
def _chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


@st.cache_data(ttl=7200, show_spinner=False)
def fetch_batch(tickers: tuple, period: str) -> dict:
    """Birden çok hisseyi chunk'lar halinde ve paralel (threads=True) olarak indirir."""
    data = {}
    tickers = list(tickers)
    for chunk in _chunked(tickers, DOWNLOAD_CHUNK_SIZE):
        try:
            raw = yf.download(
                tickers=chunk, period=period, interval="1d",
                group_by="ticker", threads=True, progress=False,
                auto_adjust=True,
            )
        except Exception:
            continue

        for t in chunk:
            try:
                if len(chunk) == 1:
                    sub = raw
                else:
                    if not isinstance(raw.columns, pd.MultiIndex) or t not in raw.columns.get_level_values(0):
                        continue
                    sub = raw[t]
                sub = sub.dropna(subset=["Close", "Volume"])
                if not sub.empty:
                    data[t] = sub
            except Exception:
                continue
    return data


@st.cache_data(ttl=7200, show_spinner=False)
def fetch_index(period: str):
    """BIST100 endeks kapanış serisini indirir (piyasa rejimi ve göreli güç için)."""
    try:
        df = yf.download(INDEX_TICKER, period=period, interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
        return df["Close"] if not df.empty else None
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Teknik gösterge temel fonksiyonları (tamamı vektörize — tüm geçmiş için hızlı)
# ----------------------------------------------------------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _wilder_rma(x: np.ndarray, period: int) -> np.ndarray:
    """Wilder'ın ORİJİNAL özyinelemeli düzleştirmesi (RMA).

    pandas'ın ewm(alpha=1/period, adjust=False) yaklaşıklamasından farkı:
    ewm ilk veri noktasından itibaren üstel ağırlıklandırma uygular ve kısa
    geçmişlerde (örn. 6 aylık tarama) gerçek Wilder tanımından sapar (ilk
    ~60-80 barda birkaç puana varan fark). Bu fonksiyon klasik tanımla
    BİREBİR eşleşir: önce ilk `period` değerin basit ortalamasıyla başlar,
    sonrasında prev*(period-1)/period + x/period ile günceller. Zincirleme
    kullanımlarda (örn. ADX içinde DX'in kendisinin tekrar düzleştirilmesi)
    girdi serisinin baştaki NaN'larını otomatik atlar.
    """
    n = len(x)
    out = np.full(n, np.nan)
    valid = ~np.isnan(x)
    if not valid.any():
        return out
    s = int(np.argmax(valid))  # ilk geçerli (NaN olmayan) indeks
    if s + period > n:
        return out
    seed = np.nanmean(x[s:s + period])
    seed_idx = s + period - 1
    out[seed_idx] = seed
    prev = seed
    for i in range(seed_idx + 1, n):
        xi = x[i]
        if np.isnan(xi):
            out[i] = prev  # ender rastlanan eksik veri: son değeri taşı
            continue
        prev = (prev * (period - 1) + xi) / period
        out[i] = prev
    return out


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder'ın orijinal (sabit-periyot basit ortalama ile başlatılan,
    özyinelemeli) RSI tanımı — yaklaşıklama değil, tam formül."""
    delta = series.diff()
    gain = delta.clip(lower=0).to_numpy()
    loss = (-delta.clip(upper=0)).to_numpy()
    avg_gain = _wilder_rma(gain, period)
    avg_loss = _wilder_rma(loss, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / np.where(avg_loss == 0, np.nan, avg_loss)
        out = 100 - (100 / (1 + rs))
    out = pd.Series(out, index=series.index)
    # Kayıp ortalaması tam sıfırsa (kesintisiz yükseliş) RSI=100 olmalı, NaN değil
    pure_uptrend = (avg_loss == 0) & ~np.isnan(avg_gain) & (avg_gain > 0)
    out[pure_uptrend] = 100.0
    return out.fillna(50)


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def wilson_interval(n_positive: int, n_total: int, z: float = 1.96) -> tuple:
    """Bir oranın (örn. tarihsel başarı oranı) %95 Wilson güven aralığını
    hesaplar. Küçük örneklemlerde basit normal yaklaşıklıktan çok daha
    güvenilirdir ve negatif/100'ü aşan aralık üretmez. (lo%, hi%) döner."""
    if n_total <= 0:
        return (np.nan, np.nan)
    phat = n_positive / n_total
    denom = 1 + z ** 2 / n_total
    center = phat + z ** 2 / (2 * n_total)
    margin = z * np.sqrt((phat * (1 - phat) + z ** 2 / (4 * n_total)) / n_total)
    lo = max(0.0, (center - margin) / denom) * 100
    hi = min(1.0, (center + margin) / denom) * 100
    return (lo, hi)


def detect_data_quality_issue(df: pd.DataFrame, threshold_pct: float = SUSPICIOUS_JUMP_PCT):
    """BIST'te tek günlük fiyat hareketi fiili olarak ~%10 bandıyla sınırlı
    olduğundan, bundan çok daha büyük tek-günlük bir sıçrama genellikle
    gerçek bir fiyat hareketi değil, yfinance'in doğru geriye-ayarlamadığı
    bir bedelsiz sermaye artırımı / bölünme gibi kurumsal bir işlemin izidir.
    Böyle bir sıçrama EMA/RSI/ADX gibi tüm göstergeleri o noktadan itibaren
    bozar; bu yüzden böyle hisseler analiz dışı bırakılır. (var_mı, sebep) döner.
    """
    chg = df["Close"].pct_change().abs() * 100
    bad = chg[chg > threshold_pct]
    if not bad.empty:
        worst_date = bad.idxmax()
        worst_val = bad.max()
        return True, (
            f"Şüpheli fiyat sıçraması: {worst_date.date()} tarihinde %{worst_val:.0f} "
            "tek günlük değişim (olası ayarlanmamış bedelsiz/bölünme — veri güvenilmez)"
        )
    return False, None


def bollinger_bandwidth(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return (upper - lower) / sma


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """Wilder'ın ADX / +DI / -DI göstergesi — _wilder_rma ile tam (yaklaşık
    değil) klasik tanım. +DI/-DI oranı hem TR hem DM aynı şekilde
    düzleştirildiği için ortaya sum-bazlı ya da average-bazlı Wilder
    smoothing kullanılması fark etmez (oranda sabit çarpan sadeleşir); ADX
    çizgisi ise DX serisinin kendisinin tekrar Wilder-ortalamasıdır."""
    idx = high.index
    high_v, low_v, close_v = high.to_numpy(), low.to_numpy(), close.to_numpy()

    up_move = np.diff(high_v, prepend=np.nan)
    down_move = -np.diff(low_v, prepend=np.nan)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = np.nan
    minus_dm[0] = np.nan

    prev_close = np.roll(close_v, 1)
    prev_close[0] = np.nan
    tr = np.nanmax(np.vstack([
        high_v - low_v,
        np.abs(high_v - prev_close),
        np.abs(low_v - prev_close),
    ]), axis=0)
    tr[0] = np.nan

    atr = _wilder_rma(tr, period)
    plus_dm_s = _wilder_rma(plus_dm, period)
    minus_dm_s = _wilder_rma(minus_dm, period)

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100 * plus_dm_s / np.where(atr == 0, np.nan, atr)
        minus_di = 100 * minus_dm_s / np.where(atr == 0, np.nan, atr)
        denom = plus_di + minus_di
        dx = 100 * np.abs(plus_di - minus_di) / np.where(denom == 0, np.nan, denom)

    adx_line = _wilder_rma(dx, period)
    return pd.Series(adx_line, index=idx), pd.Series(plus_di, index=idx), pd.Series(minus_di, index=idx)


# ----------------------------------------------------------------------------
# Analiz motoru — tüm göstergeleri ve kriterleri TÜM GEÇMİŞ için vektörize
# hesaplar. Böylece hem "bugünkü" puan hem de tarihsel backtest aynı,
# tutarlı formülden üretilir.
# ----------------------------------------------------------------------------
def build_indicator_set(
    df: pd.DataFrame,
    index_close: pd.Series = None,
    adx_threshold: int = 20,
    trigger_buffer_pct: float = DEFAULT_TRIGGER_BUFFER_PCT,
) -> dict:
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    rsi14 = rsi(close, 14)
    macd_line, signal_line, hist = macd(close)
    bw = bollinger_bandwidth(close)
    adx14, plus_di, minus_di = adx(high, low, close, 14)

    # 1. Trend: boğa dizilişi (sadece fiyat değil, EMA'ların sıralaması da)
    trend = (close > ema20) & (close > ema50) & (ema20 > ema50)

    # 5. Seviye: direnç kırılımı / yakınlığı ya da sıkışma sonrası
    # NOT: direnç artık kapanış değil, GÜNLÜK EN YÜKSEK (High) fiyatlara göre
    # hesaplanıyor — gün içi fitilleri de hesaba katan, daha standart bir
    # direnç tanımı (sadece kapanışlara bakmak gerçek direnci olduğundan
    # düşük gösterebilirdi). Momentum kriteriyle tutarlılık için bu blok
    # Trend'den hemen sonra, Momentum'dan ÖNCE hesaplanıyor (aşağıya bkz).
    recent_high20 = high.shift(1).rolling(20).max()
    breakout = close >= recent_high20
    near_resistance = close >= recent_high20 * 0.98
    bw_q20 = bw.rolling(60, min_periods=30).quantile(0.2)
    squeeze = bw <= bw_q20
    level_ok = breakout | near_resistance | squeeze

    # 2. RSI momentum: iki ayrı senaryoyu kapsar —
    #   a) "Erken/orta aşama" momentum: RSI 40-65 bandında ve yükseliyor
    #      (henüz aşırı alım değil, trend yeni güçleniyor).
    #   b) "Kırılım devamı" momentum: Seviye kriteri (breakout/direnç
    #      yakınlığı/sıkışma) zaten sağlanmışsa ve RSI 65-80 arasında
    #      yükseliyorsa da say. Gerçek kırılımlarda RSI doğal olarak 65'in
    #      üzerine çıkar; eski kodda bu durum ceza görmeden sadece "momentum"
    #      puanını kaybediyordu — bu da Seviye kriterini ödüllendirirken
    #      Momentum kriterini aynı anda cezalandıran içsel bir çelişkiydi.
    #      80 üzeri hâlâ aşırı-alım sayılıp hariç tutulur.
    rsi_prev3 = rsi14.shift(3)
    rsi_rising = rsi14 > rsi_prev3
    momentum_core = (rsi14 >= 40) & (rsi14 <= 65) & rsi_rising
    momentum_breakout = level_ok & (rsi14 > 65) & (rsi14 <= RSI_BREAKOUT_UPPER) & rsi_rising
    momentum = momentum_core | momentum_breakout

    # 3. MACD: sinyal üzerinde ve histogram güçleniyor
    hist_prev = hist.shift(1)
    macd_ok = (macd_line > signal_line) & (hist > hist_prev)

    # 4. Hacim: bugünü hariç tutan 20g ortalamanın 1.3x üzeri + fiyat yukarı
    vol_avg20 = volume.shift(1).rolling(20).mean()
    vol_ratio = volume / vol_avg20.replace(0, np.nan)
    price_up = close > close.shift(1)
    volume_ok = (vol_ratio > 1.3) & price_up

    # Giriş tetik seviyesi: "dünkü direncin birazı üzeri" — eğer hisse bugün zaten
    # kırılım yaptıysa (close >= recent_high20), referans olarak bugünün kapanışını
    # alır (devam/sürdürme teyidi istenir); henüz kırmadıysa referans direncin
    # kendisidir. İkisinin büyük olanı + küçük bir tampon = yarın izlenecek fiyat.
    # Tampon, farklı fiyat seviyelerindeki yüzlerce hissede tutarlı kalması için
    # sabit kuruş yerine yüzde olarak uygulanır.
    reference_level = pd.concat([recent_high20, close], axis=1).max(axis=1)
    entry_trigger = reference_level * (1 + trigger_buffer_pct / 100)

    # 6. ADX: trend gerçekten güçlü mü (yatay/sahte kırılımları ele)
    adx_ok = adx14 >= adx_threshold

    # 7. Göreli güç: endeksten daha iyi performans (son 20 gün)
    index_available = index_close is not None and len(index_close) > 25
    if index_available:
        idx_aligned = index_close.reindex(close.index, method="ffill")
        stock_ret20 = close / close.shift(20) - 1
        index_ret20 = idx_aligned / idx_aligned.shift(20) - 1
        rel_strength = (stock_ret20 - index_ret20) * 100
    else:
        rel_strength = pd.Series(np.nan, index=close.index)
    rs_ok = rel_strength > 0

    # Risk bayrakları
    overextended = close > ema20 * (1 + OVEREXTENSION_PCT)
    chg_pct = close.pct_change() * 100
    near_limit = chg_pct.abs() >= NEAR_LIMIT_PCT

    # Likidite (TL ciro)
    turnover = close * volume
    avg_turnover20 = turnover.rolling(20).mean()

    criteria = {
        "Trend: Fiyat > EMA20 > EMA50 (boğa dizilişi)": trend,
        "RSI(14) 40-65 yükseliyor (veya kırılımda 65-80 yükseliyor)": momentum,
        "MACD sinyal üzerinde, histogram güçleniyor": macd_ok,
        "Hacim: 20g ort. 1.3x üzeri + fiyat yukarı": volume_ok,
        "Seviye: direnç kırılımı/yakınlığı ya da sıkışma (High bazlı)": level_ok,
        f"ADX(14) ≥ {adx_threshold} (trend gerçekten güçlü)": adx_ok,
        "BIST100'den güçlü (20g göreli getiri > 0)": rs_ok,
    }
    risk_flags = {
        f"Aşırı uzamış (EMA20'nin %{OVEREXTENSION_PCT*100:.0f}+ üzerinde)": overextended,
        f"Bugün marj sınırına yakın (±%{NEAR_LIMIT_PCT:.0f}+)": near_limit,
    }

    raw_score = sum(c.astype(int) for c in criteria.values())
    penalty = sum(r.astype(int) for r in risk_flags.values())
    score_series = (raw_score - penalty).clip(lower=0)

    return {
        "close": close,
        "score_series": score_series,
        "criteria": criteria,
        "risk_flags": risk_flags,
        "ema20": ema20, "ema50": ema50, "rsi14": rsi14,
        "adx14": adx14, "rel_strength": rel_strength,
        "vol_ratio": vol_ratio, "recent_high20": recent_high20,
        "chg_pct": chg_pct, "avg_turnover20": avg_turnover20,
        "reference_level": reference_level, "entry_trigger": entry_trigger,
        "index_available": index_available,
    }


def evaluate_today(ind: dict) -> tuple:
    """build_indicator_set() çıktısından son günün puan/kriter/metriklerini çıkarır."""
    idx = -1

    def _safe(series):
        v = series.iloc[idx]
        return float(v) if pd.notna(v) else np.nan

    details = {name: bool(series.iloc[idx]) for name, series in ind["criteria"].items()}
    risk = {name: bool(series.iloc[idx]) for name, series in ind["risk_flags"].items()}
    score = int(ind["score_series"].iloc[idx])

    metrics = {
        "Kapanış": _safe(ind["close"]),
        "Değişim %": _safe(ind["chg_pct"]) if pd.notna(ind["chg_pct"].iloc[idx]) else 0.0,
        "RSI(14)": _safe(ind["rsi14"]),
        "ADX(14)": _safe(ind["adx14"]),
        "EMA20": _safe(ind["ema20"]),
        "EMA50": _safe(ind["ema50"]),
        "Hacim Oranı": _safe(ind["vol_ratio"]),
        "Gör. Güç 20g %": _safe(ind["rel_strength"]),
        "Ort. Ciro (M TL)": _safe(ind["avg_turnover20"]) / 1e6 if pd.notna(ind["avg_turnover20"].iloc[idx]) else np.nan,
        "Referans Direnç": _safe(ind["reference_level"]),
        "Giriş Tetik Fiyatı": _safe(ind["entry_trigger"]),
        "Tetiğe Uzaklık %": (
            (_safe(ind["entry_trigger"]) / _safe(ind["close"]) - 1) * 100
            if pd.notna(ind["entry_trigger"].iloc[idx]) and _safe(ind["close"]) else np.nan
        ),
        "Azami Puan": MAX_POSITIVE_CRITERIA if ind.get("index_available", True) else MAX_POSITIVE_CRITERIA - 1,
    }
    return score, details, risk, metrics


def backtest_hit_rate(ind: dict, threshold: int, min_signals: int = MIN_BACKTEST_SIGNALS) -> dict:
    """Bu hissenin kendi geçmişinde, aynı puan eşiğini geçtiği günlerden sonraki
    günün getiri istatistiklerini hesaplar (vektörize, hızlı). Tek hisseli
    örneklemler küçük olduğundan, nokta tahminine ek olarak %95 Wilson güven
    aralığı da döndürülür — böylece örn. "n=6, %83" gibi aslında çok geniş
    belirsizlik taşıyan bir sonuç yanıltıcı kesinlikte sunulmaz."""
    close = ind["close"]
    fwd = close.pct_change().shift(-1) * 100
    tmp = pd.DataFrame({"score": ind["score_series"], "fwd": fwd}).dropna()
    hits = tmp.loc[tmp["score"] >= threshold, "fwd"]
    n = len(hits)
    if n < min_signals:
        return {"n": n, "hit_rate": np.nan, "avg_return": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    n_pos = int((hits > 0).sum())
    ci_lo, ci_hi = wilson_interval(n_pos, n)
    return {
        "n": n, "hit_rate": float(n_pos / n * 100), "avg_return": float(hits.mean()),
        "ci_lo": ci_lo, "ci_hi": ci_hi,
    }


def aggregate_backtest_hit_rate(details_map: dict, threshold: int,
                                 min_signals: int = MIN_AGGREGATE_BACKTEST_SIGNALS) -> dict:
    """TÜM taranan hisselerin geçmişini birleştirip (cross-sectional / tüm
    evren bazlı), puanı eşiği geçen her hisse-gün için ertesi gün getirisini
    tek bir havuzda toplar. Tek hisse bazlı backtest'teki en büyük zayıflık
    (n genelde 5-20 gibi çok küçük bir sayı) burada ortadan kalkar: yüzlerce
    hisse x yüzlerce gün birleşince örneklem çok daha büyük ve puanlama
    sisteminin GENEL geçerliliği hakkında istatistiksel olarak daha anlamlı
    bir gösterge elde edilir. Yine de bu sinyaller zaman ve hisseler arasında
    bağımsız değildir (aynı gün birçok hisse birlikte hareket edebilir), bu
    yüzden Wilson aralığı yine de temkinli yorumlanmalıdır."""
    pooled_frames = []
    for ticker, payload in details_map.items():
        ind = payload[3]
        close = ind["close"]
        fwd = close.pct_change().shift(-1) * 100
        tmp = pd.DataFrame({"score": ind["score_series"], "fwd": fwd}).dropna()
        hits = tmp.loc[tmp["score"] >= threshold, "fwd"]
        if not hits.empty:
            pooled_frames.append(hits)

    n_stocks_with_signal = len(pooled_frames)
    if not pooled_frames:
        return {"n": 0, "n_stocks": 0, "hit_rate": np.nan, "avg_return": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}

    pooled = pd.concat(pooled_frames)
    n = len(pooled)
    if n < min_signals:
        return {"n": n, "n_stocks": n_stocks_with_signal, "hit_rate": np.nan, "avg_return": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan}

    n_pos = int((pooled > 0).sum())
    ci_lo, ci_hi = wilson_interval(n_pos, n)
    return {
        "n": n, "n_stocks": n_stocks_with_signal, "hit_rate": float(n_pos / n * 100),
        "avg_return": float(pooled.mean()), "ci_lo": ci_lo, "ci_hi": ci_hi,
    }


def score_color(val) -> str:
    """'Puan' sütunu için matplotlib gerektirmeyen hafif bir arka plan rengi üretir."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    v = max(0.0, min(MAX_POSITIVE_CRITERIA, v))
    alpha = 0.10 + (v / MAX_POSITIVE_CRITERIA) * 0.55
    return f"background-color: rgba(22, 163, 74, {alpha:.2f})"


# ----------------------------------------------------------------------------
# Grafik
# ----------------------------------------------------------------------------
def build_chart(df: pd.DataFrame, ind: dict, ticker: str) -> go.Figure:
    close = df["Close"]

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, row_heights=[0.45, 0.15, 0.2, 0.2],
        vertical_spacing=0.035, subplot_titles=(ticker, "Hacim", "RSI(14)", "ADX(14)"),
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Fiyat", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ind["ema20"], name="EMA20", line=dict(width=1.3, color="#3b82f6")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ind["ema50"], name="EMA50", line=dict(width=1.3, color="#f97316")), row=1, col=1)

    trigger_price = ind["entry_trigger"].iloc[-1]
    if pd.notna(trigger_price):
        fig.add_hline(
            y=float(trigger_price), line_dash="dash", line_color="#d97706", line_width=2,
            annotation_text=f"Giriş Tetik: {trigger_price:.2f}", annotation_position="top left",
            row=1, col=1,
        )

    vol_colors = np.where(df["Close"] >= df["Open"], "#16a34a", "#dc2626")
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors, name="Hacim", showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=ind["rsi14"], name="RSI(14)", line=dict(width=1.3, color="#8b5cf6")), row=3, col=1)
    fig.add_hline(y=65, line_dash="dot", line_color="gray", row=3, col=1)
    fig.add_hline(y=40, line_dash="dot", line_color="gray", row=3, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=ind["adx14"], name="ADX(14)", line=dict(width=1.3, color="#0f766e")), row=4, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="gray", row=4, col=1)

    fig.update_layout(height=780, margin=dict(l=10, r=10, t=40, b=10), xaxis_rangeslider_visible=False)
    return fig


# ----------------------------------------------------------------------------
# Arayüz
# ----------------------------------------------------------------------------
st.title("📈 BIST Akşam Taraması — Faz 1 (v2)")
st.caption(
    "Piyasa kapandıktan sonra çalıştırılır. Amaç: ertesi gün yükseliş potansiyeli yüksek "
    "görünen hisselerden, piyasa bağlamı ve tarihsel geçerliliğiyle desteklenmiş bir "
    "izleme listesi (watchlist) çıkarmak."
)
st.info(
    "⚠️ Bu araç yatırım tavsiyesi değildir; sadece teknik kriterlere ve geçmiş veriye dayalı "
    "bir ön-eleme/tarama aracıdır. Tarihsel başarı oranı gelecekteki sonucu garanti etmez.",
    icon="⚠️",
)

full_list = load_full_bist_list()

with st.sidebar:
    st.header("Ayarlar")

    source = st.radio(
        "Hisse listesi kaynağı",
        [f"Tüm BIST Hisseleri ({len(full_list)})", "Örnek alt küme (BIST30)", "Kendi listem"],
    )
    if source == "Kendi listem":
        raw_input = st.text_area(
            "Hisse kodlarını virgülle ayırarak girin (.IS gerekmez)",
            value="THYAO, ASELS, SASA, FROTO",
            height=100,
        )
        tickers = [normalize_ticker(t) for t in raw_input.split(",") if t.strip()]
    elif source.startswith("Örnek"):
        tickers = [normalize_ticker(t) for t in SAMPLE_TICKERS]
        st.caption(f"{len(tickers)} hisse — hızlı deneme için küçük bir alt küme.")
    else:
        tickers = [normalize_ticker(t) for t in full_list]
        st.caption(
            f"{len(tickers)} hisse taranacak. `data/bist_tickers.txt` dosyasından okunur."
        )
        st.warning(
            "Tüm liste taraması birkaç dakika sürebilir. Sonuçlar 2 saat önbellekte tutulur.",
            icon="⏳",
        )

    period = st.selectbox("Geçmiş veri aralığı", ["6mo", "1y", "2y"], index=1)
    st.caption("Daha uzun aralık, tarihsel başarı oranı için daha güvenilir örneklem sağlar.")

    min_score = st.slider(f"Minimum puan filtresi (0-{MAX_POSITIVE_CRITERIA})", 0, MAX_POSITIVE_CRITERIA, 4)
    st.caption(
        f"{MAX_POSITIVE_CRITERIA}, endeks verisi alınabildiği taramalardaki azami puandır — "
        "endeks verisi alınamazsa tarama sonrası bu taramaya özel azami puanın 6'ya düştüğünü "
        "belirten bir uyarı üstte gösterilir."
    )

    st.subheader("Gelişmiş Filtreler")
    adx_threshold = st.slider("Minimum ADX(14) — trend gücü eşiği", 10, 40, 20)
    min_turnover_m = st.number_input("Minimum ort. günlük ciro (Milyon TL, 20g)", min_value=0.0, value=20.0, step=5.0)
    exclude_illiquid = st.checkbox("Düşük likiditeli hisseleri izleme listesinden hariç tut", value=True)
    trigger_buffer_pct = st.number_input(
        "Giriş tetik tamponu (%) — referans direncin üzerine eklenir",
        min_value=0.0, max_value=5.0, value=DEFAULT_TRIGGER_BUFFER_PCT, step=0.1,
    )
    st.caption(
        "Yarın izlenecek fiyat = (dünkü direnç ya da bugünün kapanışından büyük olanı) "
        "× (1 + tampon). Sabit kuruş yerine yüzde kullanılır çünkü hisseler çok farklı "
        "fiyat seviyelerinde işlem görüyor."
    )

    run = st.button("🔍 Taramayı Başlat", type="primary", width="stretch")

if "scan" not in st.session_state:
    st.session_state.scan = None

if run:
    with st.spinner("BIST100 endeks verisi alınıyor..."):
        index_close = fetch_index(period)

    with st.spinner(f"{len(tickers)} hisse için veri indiriliyor... (büyük listelerde birkaç dakika sürebilir)"):
        batch = fetch_batch(tuple(tickers), period)

    rows = []
    details_map = {}
    failed = []
    quality_issues = []
    index_available = index_close is not None and len(index_close) > 25

    progress = st.progress(0, text="Analiz ediliyor...")
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        df = batch.get(ticker)
        if df is None or len(df) < MIN_BARS_REQUIRED:
            failed.append((ticker, "Veri yok / yetersiz"))
        else:
            bad, reason = detect_data_quality_issue(df)
            if bad:
                quality_issues.append((ticker, reason))
                continue
            try:
                ind = build_indicator_set(df, index_close, adx_threshold, trigger_buffer_pct)
                score, details, risk, metrics = evaluate_today(ind)
                bt = backtest_hit_rate(ind, min_score)
                likidite_ok = bool(pd.notna(metrics["Ort. Ciro (M TL)"]) and metrics["Ort. Ciro (M TL)"] >= min_turnover_m)

                row = {
                    "Hisse": ticker.replace(".IS", ""),
                    "Puan": score,
                    **metrics,
                    "Likidite OK": likidite_ok,
                    "Tarihsel Başarı %": bt["hit_rate"],
                    "Sinyal Sayısı": bt["n"],
                    "Ort. Sonraki Gün Getiri %": bt["avg_return"],
                    "Başarı %95 GA": (
                        f"{bt['ci_lo']:.0f}–{bt['ci_hi']:.0f}" if pd.notna(bt.get("ci_lo", np.nan)) else "—"
                    ),
                }
                rows.append(row)
                details_map[ticker] = (details, risk, df, ind, bt)
            except Exception as e:
                failed.append((ticker, str(e)))

        if (i + 1) % max(1, total // 50) == 0 or (i + 1) == total:
            progress.progress((i + 1) / total, text=f"Analiz ediliyor... ({i+1}/{total})")

    progress.empty()

    if rows:
        results_df = pd.DataFrame(rows).sort_values(
            ["Puan", "Tarihsel Başarı %"], ascending=[False, False]
        ).reset_index(drop=True)
    else:
        results_df = pd.DataFrame()

    agg_bt = aggregate_backtest_hit_rate(details_map, min_score)

    st.session_state.scan = {
        "results": results_df,
        "details": details_map,
        "failed": failed,
        "quality_issues": quality_issues,
        "index_close": index_close,
        "index_available": index_available,
        "min_score": min_score,
        "exclude_illiquid": exclude_illiquid,
        "agg_backtest": agg_bt,
        "timestamp": datetime.now().strftime("%d.%m.%Y %H:%M"),
    }

scan = st.session_state.scan

if scan is None:
    st.write("👈 Soldan hisse listenizi ve ayarlarınızı seçip **Taramayı Başlat**'a basın.")
else:
    results_df = scan["results"]
    st.caption(f"Son tarama: {scan['timestamp']}")

    # --- Piyasa rejimi bilgi panosu ---
    # ÖNEMLİ: "azami puan" uyarısı, puanlamada GERÇEKTEN kullanılan aynı
    # `index_available` bayrağına (build_indicator_set ile aynı eşik, >25 bar)
    # dayanır — EMA50 rejim panosu için gereken ayrı ve daha sıkı (>=50 bar)
    # eşikle KARIŞTIRILMAZ. Önceki sürümde bu iki ayrı eşik aynı anda, tek bir
    # if/else içinde kullanıldığından, endeks verisi 26-49 bar gibi "Göreli
    # Güç kriterini hesaplamaya yeter ama EMA50 rejim panosu için yetmez"
    # bir aralıkta kaldığında ekranda "azami puan 6" uyarısı çıkıyor, ama
    # puanlama aslında 7 üzerinden devam ediyordu (bildirdiğiniz hata buydu).
    idx_close = scan.get("index_close")
    index_available = scan.get("index_available", False)

    if not index_available:
        st.error(
            "⚠️ BIST100 endeks verisi alınamadı! 'Göreli Güç' kriteri bu taramada hiçbir hisse için "
            f"hesaplanamadı — dolayısıyla bu taramadaki azami puan 7 değil **{MAX_POSITIVE_CRITERIA - 1}** "
            "olarak değerlendirilmelidir. Puanları buna göre yorumlayın veya taramayı tekrar deneyin.",
            icon="⚠️",
        )
    elif len(idx_close) >= 50:
        idx_ema50 = ema(idx_close, 50)
        idx_up = bool(idx_close.iloc[-1] > idx_ema50.iloc[-1])
        idx_chg = float((idx_close.iloc[-1] / idx_close.iloc[-2] - 1) * 100) if len(idx_close) > 1 else 0.0
        if idx_up:
            st.success(
                f"📊 BIST100 endeksi EMA50 üzerinde (genel eğilim yukarı yönlü). "
                f"Son kapanış değişimi: {idx_chg:+.2f}%",
            )
        else:
            st.warning(
                f"📊 BIST100 endeksi EMA50 altında (genel eğilim zayıf/aşağı yönlü) — "
                f"bireysel al sinyallerinin güvenilirliği böyle ortamlarda genelde düşer. "
                f"Son kapanış değişimi: {idx_chg:+.2f}%",
            )
    else:
        # Göreli Güç kriteri hesaplanabildi (azami puan hâlâ 7) ama endeks
        # geçmişi EMA50 bazlı piyasa rejimi notu için yetersiz (<50 bar).
        st.caption(
            f"ℹ️ Endeks geçmişi sınırlı ({len(idx_close)} bar): Göreli Güç kriteri hesaplanabildi "
            "(azami puan hâlâ 7), ancak EMA50 bazlı piyasa rejimi notu için yeterli veri yok."
        )

    # --- Tüm evren bazlı (cross-sectional) tarihsel doğrulama ---
    agg_bt = scan.get("agg_backtest")
    if agg_bt is not None:
        if pd.notna(agg_bt.get("hit_rate", np.nan)):
            st.info(
                f"📐 **Tüm evren bazlı tarihsel doğrulama** (puan ≥ {scan['min_score']}, "
                f"{agg_bt['n_stocks']} farklı hissenin geçmişi birleştirilerek): "
                f"**{agg_bt['n']}** hisse-gün sinyali bulundu, bunların **%{agg_bt['hit_rate']:.0f}'i** "
                f"ertesi gün pozitif kapandı (%95 güven aralığı: **{agg_bt['ci_lo']:.0f}–{agg_bt['ci_hi']:.0f}**), "
                f"ortalama ertesi gün getirisi **{agg_bt['avg_return']:+.2f}%**. "
                "Tek hisse bazlı backtest'ten çok daha büyük bir örneklem olduğu için puanlama "
                "sisteminin genel geçerliliği hakkında daha güvenilir bir fikir verir — ama sinyaller "
                "aynı gün birçok hissede birlikte tetiklenebildiğinden (bağımsız değildir), yine de "
                "temkinli yorumlanmalıdır.",
                icon="📐",
            )
        else:
            st.caption(
                f"ℹ️ Tüm evren bazlı doğrulama için yeterli sinyal yok (bulunan: {agg_bt['n']}, "
                f"gereken minimum: {MIN_AGGREGATE_BACKTEST_SIGNALS}). Daha fazla hisse veya daha uzun "
                "bir veri aralığı deneyin."
            )

    if scan.get("quality_issues"):
        with st.expander(f"🚫 Veri kalitesi nedeniyle hariç tutulan {len(scan['quality_issues'])} hisse"):
            st.caption(
                "Bu hisselerde yfinance'in doğru ayarlamadığı bir bedelsiz sermaye artırımı/bölünme "
                "gibi bir kurumsal işlem izi tespit edildi; tüm göstergeler bu noktadan itibaren "
                "bozulabileceği için analiz dışı bırakıldı."
            )
            for t, reason in scan["quality_issues"]:
                st.write(f"- {t}: {reason}")

    if results_df.empty:
        st.warning("Hiçbir hisse için analiz tamamlanamadı. Hisse kodlarını ve bağlantıyı kontrol edin.")
    else:
        min_score_used = scan["min_score"]
        base_filter = results_df["Puan"] >= min_score_used
        if scan["exclude_illiquid"]:
            base_filter &= results_df["Likidite OK"]
        watchlist = results_df[base_filter]

        display_cols = [
            "Hisse", "Puan", "Kapanış", "Değişim %",
            "Giriş Tetik Fiyatı", "Tetiğe Uzaklık %", "Referans Direnç",
            "RSI(14)", "ADX(14)", "Hacim Oranı", "Gör. Güç 20g %", "Ort. Ciro (M TL)",
            "Tarihsel Başarı %", "Başarı %95 GA", "Sinyal Sayısı", "Ort. Sonraki Gün Getiri %",
        ]
        fmt = {
            "Kapanış": "{:.2f}", "Değişim %": "{:+.2f}",
            "Giriş Tetik Fiyatı": "{:.2f}", "Tetiğe Uzaklık %": "{:+.2f}", "Referans Direnç": "{:.2f}",
            "RSI(14)": "{:.1f}", "ADX(14)": "{:.1f}", "Hacim Oranı": "{:.2f}x", "Gör. Güç 20g %": "{:+.1f}",
            "Ort. Ciro (M TL)": "{:.1f}", "Tarihsel Başarı %": "{:.0f}",
            "Ort. Sonraki Gün Getiri %": "{:+.2f}",
        }

        tab1, tab2 = st.tabs(["📋 İzleme Listesi", "📊 Tüm Sonuçlar"])

        with tab1:
            st.subheader(f"Puanı ≥ {min_score_used} olan hisseler ({len(watchlist)})")
            if scan["exclude_illiquid"]:
                st.caption("Düşük likiditeli hisseler bu listeden hariç tutuldu (sol menüden değiştirilebilir).")
            if watchlist.empty:
                st.write("Bu eşiği geçen hisse yok. Eşiği düşürmeyi veya likidite filtresini gevşetmeyi deneyin.")
            else:
                st.dataframe(
                    watchlist[display_cols].style.format(fmt, na_rep="—").map(score_color, subset=["Puan"]),
                    width="stretch", hide_index=True,
                )
                csv = watchlist[display_cols].to_csv(index=False).encode("utf-8-sig")
                st.download_button("⬇️ İzleme listesini CSV olarak indir", csv, "izleme_listesi.csv", "text/csv")

        with tab2:
            st.dataframe(
                results_df[display_cols + ["Likidite OK"]].style.format(fmt, na_rep="—"),
                width="stretch", hide_index=True,
            )

        if scan["failed"]:
            with st.expander(f"⚠️ Analiz edilemeyen {len(scan['failed'])} hisse"):
                for t, reason in scan["failed"]:
                    st.write(f"- {t}: {reason}")

        st.divider()
        st.subheader("Detay İncele")
        chosen = st.selectbox("Bir hisse seçin", results_df["Hisse"].tolist())
        chosen_ticker = f"{chosen}.IS"
        if chosen_ticker in scan["details"]:
            details, risk, df, ind, bt = scan["details"][chosen_ticker]
            row = results_df[results_df["Hisse"] == chosen].iloc[0]

            azami = int(row["Azami Puan"]) if pd.notna(row.get("Azami Puan", np.nan)) else MAX_POSITIVE_CRITERIA
            st.write(f"**Puan: {int(row['Puan'])} / {azami}** (risk cezaları sonrası)")
            if azami < MAX_POSITIVE_CRITERIA:
                st.caption("ℹ️ Endeks verisi alınamadığı için azami puan bu taramada 7 değil 6.")

            trig_cols = st.columns(3)
            trig_cols[0].metric("Giriş Tetik Fiyatı", f"{row['Giriş Tetik Fiyatı']:.2f} TL")
            trig_cols[1].metric("Referans Direnç", f"{row['Referans Direnç']:.2f} TL")
            trig_cols[2].metric("Tetiğe Uzaklık", f"{row['Tetiğe Uzaklık %']:+.2f}%")
            st.caption(
                f"Yarın fiyat **{row['Giriş Tetik Fiyatı']:.2f} TL** seviyesini yukarı keser, "
                "Faz 2 kriterleri (yüksek Relative Volume + yön teyidi) sağlanırsa giriş "
                "düşünülebilir. Bu seviye otomatik emir değildir, sadece izlenecek referanstır."
            )

            with st.expander("✅ Kriter ve risk detayları", expanded=True):
                st.markdown("**Pozitif kriterler:**")
                for label, ok in details.items():
                    st.write(("✅ " if ok else "➖ ") + label)
                st.markdown("**Risk bayrakları:**")
                for label, flagged in risk.items():
                    st.write(("🚩 " if flagged else "✅ Yok — ") + label)

            if pd.notna(bt["hit_rate"]):
                st.info(
                    f"📈 **Tarihsel performans** (bu hissede puan ≥ {min_score_used} olan geçmiş günler): "
                    f"{bt['n']} sinyal bulundu, bunların **%{bt['hit_rate']:.0f}'i** ertesi gün pozitif "
                    f"kapandı (%95 güven aralığı: **{bt['ci_lo']:.0f}–{bt['ci_hi']:.0f}**), ortalama ertesi "
                    f"gün getirisi **{bt['avg_return']:+.2f}%**. Düşük sinyal sayısında bu aralık çok "
                    "geniş olabilir — tek başına karar vermek için yeterli değildir.",
                )
            else:
                st.caption(
                    f"ℹ️ Tarihsel başarı oranı için yeterli geçmiş sinyal yok (bulunan: {bt['n']}, "
                    f"gereken minimum: {MIN_BACKTEST_SIGNALS}). Daha uzun bir veri aralığı deneyin."
                )

            st.plotly_chart(build_chart(df, ind, chosen), width="stretch")
