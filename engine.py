"""
engine.py — BIST Akşam Taraması hesaplama motoru (v4)
-----------------------------------------------------
Streamlit'ten BAĞIMSIZDIR: göstergeler, puanlama, veri temizliği ve tarihsel
doğrulama (backtest) burada yaşar. Böylece `pytest` ile test edilebilir ve
arayüz (app.py) yalnızca sunumdan sorumlu kalır.

PUANLAMA — azami 7 pozitif puan, 2 risk cezası (net puan 0'ın altına inmez):

  Pozitif kriterler (her biri +1):
    1. Trend     : Fiyat > EMA20 > EMA50
    2. Momentum  : RSI(14) 40-65 ve yükseliyor (kırılımda 65-80 de kabul)
    3. MACD      : MACD > sinyal ve histogram güçleniyor
    4. Hacim     : Hacim, önceki 20g ortalamanın 1.3x üzerinde + fiyat yukarı
    5. Seviye    : 20g High-bazlı direnç kırılımı/yakınlığı ya da yükseliş yanlı sıkışma
    6. ADX       : ADX(14) >= eşik VE +DI > -DI (güç + yön)
    7. Göreli Güç: 20g getiri, BIST100'den yüksek

  Risk cezaları (her biri -1):
    A. Aşırı uzama : Fiyat, EMA20'nin %10+ üzerinde
    B. Marj sınırı : Bugünkü değişim zaten ±%9+

  AZAMI PUAN: endeks verisi varsa 7, yoksa 6 (Göreli Güç hesaplanamaz).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

APP_VERSION = "v4"

# ----------------------------------------------------------------------------
# Sabitler
# ----------------------------------------------------------------------------
MIN_BARS_REQUIRED = 80            # EMA50, ADX ve sıkışma analizi için gereken minimum bar
WARMUP_BARS = 60                  # Backtest'te göstergelerin oturması için atlanan ilk bar sayısı
MAX_POSITIVE_CRITERIA = 7         # Pozitif kriter sayısı (puanın üst sınırı)
OVEREXTENSION_PCT = 0.10          # EMA20'nin bu oranın üzerinde ise "aşırı uzamış"
NEAR_LIMIT_PCT = 9.0              # Günlük değişim bu yüzdeyi geçerse "marj sınırına yakın"
RSI_BREAKOUT_UPPER = 80           # Kırılım eşliğinde izin verilen üst RSI sınırı
DEFAULT_TRIGGER_BUFFER_PCT = 0.5  # Giriş tetik seviyesi tamponu (%)
SUSPICIOUS_JUMP_PCT = 30.0        # Tek günlük |değişim| bu eşiği geçerse "olası ayarlanmamış kurumsal işlem"
MIN_BACKTEST_SIGNALS = 5          # Tek hisse için gereken min. sinyal sayısı
MIN_AGGREGATE_BACKTEST_SIGNALS = 30  # Tüm-evren backtest için gereken min. sinyal-gün sayısı
LIMIT_UP_OPEN_PCT = 9.5           # Açılış, dünkü kapanışın bu kadar üzerindeyse (tavan açılış) alım gerçekçi değil
DEFAULT_COST_PCT = 0.10           # Tek yön işlem maliyeti (%): komisyon + kayma tahmini
STALE_MAX_LAG_DAYS = 4            # Hissenin son verisi evrenin son tarihinden bu kadar günden eskiyse "eski veri"
MARKET_CLOSE_SAFE = (18, 15)      # BIST kapanış seansı bittikten sonra güvenli saat (TSİ)

# Kural anahtarları: False yapılırsa v3 davranışına dönülür.
ADX_REQUIRE_DI_DIRECTION = True   # ADX puanı için +DI > -DI şartı (ADX yalnızca güçtür, yön değildir)
SQUEEZE_REQUIRES_BULLISH = True   # Sıkışma "Seviye" puanını yalnızca fiyat EMA20 üzerindeyken ver


# ----------------------------------------------------------------------------
# Zaman yardımcıları
# ----------------------------------------------------------------------------
def istanbul_now() -> datetime:
    """Europe/Istanbul saatiyle şimdi (zoneinfo yoksa sabit UTC+3)."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Europe/Istanbul"))
    except Exception:
        return datetime.now(timezone(timedelta(hours=3)))


# ----------------------------------------------------------------------------
# Veri hijyeni
# ----------------------------------------------------------------------------
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def normalize_index(idx: pd.Index) -> pd.DatetimeIndex:
    """Zaman dilimini kaldırıp günlüğe indirger. Hisse (tz-aware) ile endeks
    (tz-naive) indeksleri karışınca `reindex` TypeError verip TÜM hisseleri
    sessizce düşürebiliyordu; bu fonksiyon iki tarafı da aynı biçime getirir."""
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def normalize_index_series(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = normalize_index(s.index)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return pd.to_numeric(s, errors="coerce").dropna()


def clean_ohlcv(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """OHLCV çerçevesini tutarlı hale getirir: tz'siz günlük indeks, tekrarsız,
    sıralı, sayısal; Close/Volume eksik satırlar atılır, High/Low/Open eksikse Close ile doldurulur."""
    if df is None or df.empty:
        return None
    if not set(OHLCV).issubset(df.columns):
        return None
    out = df[OHLCV].apply(pd.to_numeric, errors="coerce").copy()
    out.index = normalize_index(out.index)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["Close", "Volume"])
    out = out[out["Close"] > 0]
    if out.empty:
        return None
    for col in ("Open", "High", "Low"):
        out[col] = out[col].fillna(out["Close"])
    return out


def extract_ticker_frame(raw: pd.DataFrame | None, ticker: str, n_requested: int) -> pd.DataFrame | None:
    """yfinance.download çıktısından tek hissenin OHLCV çerçevesini çıkarır.

    yfinance sürümüne göre tek hisse bile (group_by='ticker') ÇOK SEVİYELİ
    sütun döndürebiliyor; eski kod bu durumda `KeyError` ile hisseyi sessizce
    kaybediyordu (özellikle 'Kendi listem'de tek hisse girildiğinde)."""
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        lv0 = raw.columns.get_level_values(0)
        lv1 = raw.columns.get_level_values(1)
        if ticker in lv0:
            sub = raw[ticker]
        elif ticker in lv1:
            sub = raw.xs(ticker, axis=1, level=1)
        else:
            return None
    else:
        if n_requested != 1:
            return None
        sub = raw
    return clean_ohlcv(sub)


def detect_data_quality_issue(df: pd.DataFrame, threshold_pct: float = SUSPICIOUS_JUMP_PCT):
    """Tek günlük |değişim| eşiği aşarsa (BIST'te fiili ±%10 bandı) olası ayarlanmamış
    bedelsiz/bölünme izi. (var_mı, sebep) döner."""
    chg = df["Close"].pct_change().abs() * 100
    bad = chg[chg > threshold_pct]
    if not bad.empty:
        worst_date, worst_val = bad.idxmax(), bad.max()
        return True, (
            f"Şüpheli fiyat sıçraması: {worst_date.date()} tarihinde %{worst_val:.0f} "
            "tek günlük değişim (olası ayarlanmamış bedelsiz/bölünme — veri güvenilmez)"
        )
    return False, None


def prepare_frame(df: pd.DataFrame, min_bars: int = MIN_BARS_REQUIRED):
    """Veri kalitesi sorunlu hisseyi tümden atmak yerine, SON şüpheli sıçramadan
    sonrasını kullanır (yeterli bar kalıyorsa). Geriye (df|None, not|None, sebep|None) döner.
      - Sorun yok            -> (df, None, None)
      - Sıçrama eskiydi      -> (kırpılmış df, açıklama, None)
      - Kırpınca yetersiz    -> (None, None, sebep)"""
    bad, reason = detect_data_quality_issue(df)
    if not bad:
        return df, None, None
    chg = df["Close"].pct_change().abs() * 100
    last_bad = chg[chg > SUSPICIOUS_JUMP_PCT].index[-1]
    pos = df.index.get_loc(last_bad)
    trimmed = df.iloc[pos:]
    if len(trimmed) >= min_bars:
        note = (f"{last_bad.date()} tarihindeki şüpheli sıçrama nedeniyle geçmiş kırpıldı; "
                f"{len(trimmed)} bar kullanıldı")
        return trimmed, note, None
    return None, None, reason


def staleness_reason(df: pd.DataFrame, ref_date: pd.Timestamp, max_lag_days: int = STALE_MAX_LAG_DAYS):
    """İşlem görmeyen/askıdaki ya da verisi eski kalmış hisseyi yakalar."""
    last = df.index[-1]
    if (ref_date - last).days > max_lag_days:
        return f"Son veri {last.date()} — işlem görmüyor ya da veri eski"
    if float(df["Volume"].iloc[-1]) == 0:
        return "Son gün hacim 0 — işlem yapılmamış"
    return None


def drop_incomplete_last_bar(df: pd.DataFrame, now: datetime | None = None):
    """Seans bitmeden çalıştırılırsa yfinance'in bugünkü YARIM mumunu çıkarır
    (yarım mumla hesaplanan RSI/MACD/hacim oranı yanıltıcıdır). (df, çıkarıldı_mı) döner."""
    now = now or istanbul_now()
    if df is None or df.empty:
        return df, False
    last = df.index[-1].date()
    session_open_day = now.weekday() < 5
    before_close = (now.hour, now.minute) < MARKET_CLOSE_SAFE
    if session_open_day and before_close and last == now.date():
        return df.iloc[:-1], True
    return df, False


def index_usable(index_close: pd.Series | None, ref_date: pd.Timestamp | None = None) -> bool:
    """Göreli Güç kriteri için endeks serisi yeterli ve güncel mi?"""
    if index_close is None or len(index_close) <= 25:
        return False
    if ref_date is not None and (ref_date - index_close.index[-1]).days > 5:
        return False
    return True


# ----------------------------------------------------------------------------
# Teknik göstergeler (vektörize)
# ----------------------------------------------------------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _wilder_rma(x: np.ndarray, period: int) -> np.ndarray:
    """Wilder'ın orijinal özyinelemeli düzleştirmesi: ilk `period` değerin
    basit ortalamasıyla başlar, sonra prev*(p-1)/p + x/p ile ilerler.
    Baştaki NaN'lar atlanır (zincirleme kullanım için)."""
    n = len(x)
    out = np.full(n, np.nan)
    valid = ~np.isnan(x)
    if not valid.any():
        return out
    s = int(np.argmax(valid))
    if s + period > n:
        return out
    seed = np.nanmean(x[s:s + period])
    seed_idx = s + period - 1
    out[seed_idx] = seed
    prev = seed
    for i in range(seed_idx + 1, n):
        xi = x[i]
        if np.isnan(xi):
            out[i] = prev
            continue
        prev = (prev * (period - 1) + xi) / period
        out[i] = prev
    return out


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI. Isınma dönemi NaN kalır (v3'te 50'ye dolduruluyordu; bu,
    erken barlarda sahte 'sakin RSI' değeri üretiyordu). Hiç kayıp yoksa 100,
    hiç hareket yoksa 50."""
    delta = series.diff()
    gain = delta.clip(lower=0).to_numpy()
    loss = (-delta.clip(upper=0)).to_numpy()
    avg_gain = _wilder_rma(gain, period)
    avg_loss = _wilder_rma(loss, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / np.where(avg_loss == 0, np.nan, avg_loss)
        out = 100 - (100 / (1 + rs))
    out = np.where((avg_loss == 0) & (avg_gain > 0), 100.0, out)
    out = np.where((avg_loss == 0) & (avg_gain == 0), 50.0, out)
    out = np.where(np.isnan(avg_gain) | np.isnan(avg_loss), np.nan, out)
    return pd.Series(out, index=series.index)


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def bollinger_bandwidth(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (2 * num_std * std) / sma


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """Wilder ADX / +DI / -DI (tam klasik tanım, _wilder_rma ile)."""
    idx = high.index
    high_v, low_v, close_v = high.to_numpy(float), low.to_numpy(float), close.to_numpy(float)

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
# İstatistik yardımcıları
# ----------------------------------------------------------------------------
def wilson_interval_vec(n_pos, n_total, z: float = 1.96):
    """%95 Wilson aralığı (vektörel). (lo%, hi%) dizileri döner; n=0 -> NaN."""
    n_pos = np.asarray(n_pos, dtype=float)
    n = np.asarray(n_total, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        phat = n_pos / n
        denom = 1 + z ** 2 / n
        center = phat + z ** 2 / (2 * n)
        margin = z * np.sqrt((phat * (1 - phat) + z ** 2 / (4 * n)) / n)
        lo = np.clip((center - margin) / denom, 0, 1) * 100
        hi = np.clip((center + margin) / denom, 0, 1) * 100
    lo = np.where(n > 0, lo, np.nan)
    hi = np.where(n > 0, hi, np.nan)
    return lo, hi


def wilson_interval(n_positive: int, n_total: int, z: float = 1.96) -> tuple:
    if n_total <= 0:
        return (np.nan, np.nan)
    lo, hi = wilson_interval_vec(n_positive, n_total, z)
    return (float(lo), float(hi))


def cluster_mean_ci(values: np.ndarray, groups: np.ndarray, z: float = 1.96) -> tuple:
    """Ortalama için GÜN-KÜMELENMİŞ (cluster-robust) %95 güven aralığı.

    Aynı gün birçok hissede birlikte sinyal çıkar ve bu gözlemler bağımsız
    değildir (piyasa yönü ortaktır). Düz Wilson aralığı bu yüzden tüm-evren
    backtest'te fazla DAR çıkar; burada gruplar (günler) birim alınır."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return (np.nan, np.nan)
    mean = x.mean()
    sums = pd.Series(x - mean).groupby(np.asarray(groups)).sum()
    g = len(sums)
    if g < 2:
        return (np.nan, np.nan)
    var = float((sums ** 2).sum()) / n ** 2 * g / (g - 1)
    se = var ** 0.5
    return (mean - z * se, mean + z * se)


# ----------------------------------------------------------------------------
# Analiz motoru
# ----------------------------------------------------------------------------
def trigger_forward_return(df: pd.DataFrame, entry_trigger: pd.Series):
    """Giriş-tetik bazlı gerçekçi ertesi gün getirisi (brüt, % cinsinden).

    Sinyal günü t'nin kapanışında karar verilir; t+1'de fiyat `entry_trigger`
    seviyesine ulaşırsa (High >= tetik) alım yapılmış sayılır:
      - Açılış tetiğin üzerindeyse (gap) açılıştan,
      - değilse tetik fiyatından dolum,
      - çıkış t+1 kapanışında.
    Açılış tavana yakınsa (>= %9.5 gap) alım gerçekçi olmadığı için hariç tutulur.
    Dönen seri: işlem gerçekleşmediyse NaN."""
    nxt_open = df["Open"].shift(-1)
    nxt_high = df["High"].shift(-1)
    nxt_close = df["Close"].shift(-1)
    gap_pct = (nxt_open / df["Close"] - 1) * 100
    triggered = (nxt_high >= entry_trigger) & nxt_close.notna()
    tradable = triggered & (gap_pct < LIMIT_UP_OPEN_PCT)
    fill = np.maximum(nxt_open, entry_trigger)
    ret = (nxt_close / fill - 1) * 100
    return ret.where(tradable), triggered


def build_indicator_set(
    df: pd.DataFrame,
    index_close: pd.Series | None = None,
    adx_threshold: int = 20,
    trigger_buffer_pct: float = DEFAULT_TRIGGER_BUFFER_PCT,
) -> dict:
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    ema20, ema50 = ema(close, 20), ema(close, 50)
    rsi14 = rsi(close, 14)
    macd_line, signal_line, hist = macd(close)
    bw = bollinger_bandwidth(close)
    adx14, plus_di, minus_di = adx(high, low, close, 14)

    # 1. Trend
    trend = (close > ema20) & (close > ema50) & (ema20 > ema50)

    # 5. Seviye (High bazlı direnç). Momentum'dan önce hesaplanır (kırılım RSI bandını etkiler).
    recent_high20 = high.shift(1).rolling(20).max()
    breakout = close >= recent_high20
    near_resistance = close >= recent_high20 * 0.98
    bw_q20 = bw.rolling(60, min_periods=30).quantile(0.2)
    squeeze = bw <= bw_q20
    if SQUEEZE_REQUIRES_BULLISH:
        squeeze = squeeze & (close > ema20)   # yönsüz sıkışma boğa sinyali değildir
    level_ok = breakout | near_resistance | squeeze

    # 2. Momentum
    rsi_rising = rsi14 > rsi14.shift(3)
    momentum_core = (rsi14 >= 40) & (rsi14 <= 65) & rsi_rising
    momentum_breakout = level_ok & (rsi14 > 65) & (rsi14 <= RSI_BREAKOUT_UPPER) & rsi_rising
    momentum = momentum_core | momentum_breakout

    # 3. MACD
    macd_ok = (macd_line > signal_line) & (hist > hist.shift(1))

    # 4. Hacim (bugün hariç 20g ortalama)
    vol_avg20 = volume.shift(1).rolling(20).mean()
    vol_ratio = volume / vol_avg20.replace(0, np.nan)
    volume_ok = (vol_ratio > 1.3) & (close > close.shift(1))

    # Giriş tetiği
    reference_level = pd.concat([recent_high20, close], axis=1).max(axis=1)
    entry_trigger = reference_level * (1 + trigger_buffer_pct / 100)

    # 6. ADX: güç + yön
    adx_ok = adx14 >= adx_threshold
    if ADX_REQUIRE_DI_DIRECTION:
        adx_ok = adx_ok & (plus_di > minus_di)

    # 7. Göreli güç (endeks tz/tarih uyumu garanti altına alınır)
    rel_strength = pd.Series(np.nan, index=close.index)
    index_available = False
    if index_close is not None and len(index_close) > 25:
        idx_norm = normalize_index_series(index_close)
        idx_aligned = idx_norm.reindex(close.index, method="ffill")
        fresh = len(idx_norm) > 0 and (close.index[-1] - idx_norm.index[-1]).days <= 5
        if fresh and idx_aligned.notna().sum() > 25:
            index_available = True
            stock_ret20 = close / close.shift(20) - 1
            index_ret20 = idx_aligned / idx_aligned.shift(20) - 1
            rel_strength = (stock_ret20 - index_ret20) * 100
    rs_ok = rel_strength > 0

    # Risk bayrakları
    overextended = close > ema20 * (1 + OVEREXTENSION_PCT)
    chg_pct = close.pct_change() * 100
    near_limit = chg_pct.abs() >= NEAR_LIMIT_PCT

    # Likidite
    avg_turnover20 = (close * volume).rolling(20).mean()

    criteria = {
        "Trend: Fiyat > EMA20 > EMA50 (boğa dizilişi)": trend,
        "RSI(14) 40-65 yükseliyor (veya kırılımda 65-80 yükseliyor)": momentum,
        "MACD sinyal üzerinde, histogram güçleniyor": macd_ok,
        "Hacim: 20g ort. 1.3x üzeri + fiyat yukarı": volume_ok,
        "Seviye: direnç kırılımı/yakınlığı ya da yükseliş yanlı sıkışma (High bazlı)": level_ok,
        (f"ADX(14) ≥ {adx_threshold} ve +DI > −DI (yönlü trend gücü)"
         if ADX_REQUIRE_DI_DIRECTION else f"ADX(14) ≥ {adx_threshold} (trend gücü)"): adx_ok,
        "BIST100'den güçlü (20g göreli getiri > 0)": rs_ok,
    }
    risk_flags = {
        f"Aşırı uzamış (EMA20'nin %{OVEREXTENSION_PCT * 100:.0f}+ üzerinde)": overextended,
        f"Bugün marj sınırına yakın (±%{NEAR_LIMIT_PCT:.0f}+)": near_limit,
    }

    raw_score = sum(c.fillna(False).astype(int) for c in criteria.values())
    penalty = sum(r.fillna(False).astype(int) for r in risk_flags.values())
    score_series = (raw_score - penalty).clip(lower=0)

    # Backtest girdileri
    fwd_cc = close.pct_change().shift(-1) * 100
    fwd_trig, triggered = trigger_forward_return(df, entry_trigger)
    has_next = close.shift(-1).notna()
    backtest_mask = pd.Series(np.arange(len(close)) >= WARMUP_BARS, index=close.index) & has_next

    return {
        "close": close,
        "score_series": score_series, "raw_score_series": raw_score, "penalty_series": penalty,
        "criteria": criteria, "risk_flags": risk_flags,
        "max_score": len(criteria) if index_available else len(criteria) - 1,
        "ema20": ema20, "ema50": ema50, "rsi14": rsi14,
        "macd_line": macd_line, "signal_line": signal_line, "macd_hist": hist,
        "adx14": adx14, "plus_di": plus_di, "minus_di": minus_di, "rel_strength": rel_strength,
        "vol_ratio": vol_ratio, "recent_high20": recent_high20,
        "chg_pct": chg_pct, "avg_turnover20": avg_turnover20,
        "reference_level": reference_level, "entry_trigger": entry_trigger,
        "index_available": index_available,
        "fwd_cc": fwd_cc, "fwd_trig": fwd_trig, "triggered": triggered,
        "backtest_mask": backtest_mask,
    }


def evaluate_today(ind: dict) -> tuple:
    """Son günün puan/kriter/metriklerini çıkarır."""
    i = -1

    def _safe(series):
        v = series.iloc[i]
        return float(v) if pd.notna(v) else np.nan

    details = {name: bool(s.iloc[i]) if pd.notna(s.iloc[i]) else False for name, s in ind["criteria"].items()}
    risk = {name: bool(s.iloc[i]) if pd.notna(s.iloc[i]) else False for name, s in ind["risk_flags"].items()}
    score = int(ind["score_series"].iloc[i])

    close_v, trig_v = _safe(ind["close"]), _safe(ind["entry_trigger"])
    turnover = _safe(ind["avg_turnover20"])
    metrics = {
        "Kapanış": close_v,
        "Değişim %": _safe(ind["chg_pct"]) if pd.notna(ind["chg_pct"].iloc[i]) else 0.0,
        "RSI(14)": _safe(ind["rsi14"]),
        "ADX(14)": _safe(ind["adx14"]),
        "EMA20": _safe(ind["ema20"]),
        "EMA50": _safe(ind["ema50"]),
        "Hacim Oranı": _safe(ind["vol_ratio"]),
        "Gör. Güç 20g %": _safe(ind["rel_strength"]),
        "Ort. Ciro (M TL)": turnover / 1e6 if pd.notna(turnover) else np.nan,
        "Referans Direnç": _safe(ind["reference_level"]),
        "Giriş Tetik Fiyatı": trig_v,
        "Tetiğe Uzaklık %": (trig_v / close_v - 1) * 100 if pd.notna(trig_v) and close_v else np.nan,
        "Ham Puan": int(ind["raw_score_series"].iloc[i]),
        "Risk Cezası": int(ind["penalty_series"].iloc[i]),
        "Azami Puan": int(ind["max_score"]),
    }
    return score, details, risk, metrics


# ----------------------------------------------------------------------------
# Tarihsel doğrulama (backtest) — "havuz" (long format) üzerinden
# ----------------------------------------------------------------------------
def pool_frame(ticker: str, ind: dict) -> pd.DataFrame:
    """Tek hissenin backtest satırları (yalnızca ısınma sonrası ve ertesi günü bilinen barlar)."""
    f = pd.DataFrame({
        "ticker": ticker,
        "score": ind["score_series"],
        "fwd_cc": ind["fwd_cc"],
        "fwd_trig": ind["fwd_trig"],
    })
    return f[ind["backtest_mask"]]


def build_pool(frames: list) -> pd.DataFrame:
    """pool_frame çıktılarını uzun formatlı havuza (index=tarih) birleştirir.
    Eşik/maliyet değişince yeniden tarama gerekmez; her şey bu havuzdan türetilir."""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=["ticker", "score", "fwd_cc", "fwd_trig"])
    pool = pd.concat(frames)
    pool.index.name = "date"
    return pool


def _empty_summary() -> dict:
    return {
        "n": 0, "n_days": 0, "n_stocks": 0,
        "hit_rate": np.nan, "avg_return": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
        "ret_ci_lo": np.nan, "ret_ci_hi": np.nan,
        "trig_n": 0, "trig_hit_rate": np.nan, "trig_avg_return": np.nan,
    }


def summarize_signals(sub: pd.DataFrame, min_signals: int, cost_pct: float = DEFAULT_COST_PCT,
                      cluster: bool = False) -> dict:
    """Bir sinyal kümesi için özet. `cluster=True` ise güven aralığı gün-kümelenmeli hesaplanır.
    İki ayrı ölçüt:
      - Kapanıştan kapanışa (brüt): t kapanışı -> t+1 kapanışı
      - Tetik bazlı (net): t+1'de tetik görülürse dolumdan t+1 kapanışına, tur başına 2×maliyet düşülür."""
    res = _empty_summary()
    n = len(sub)
    res["n"] = n
    if n == 0:
        return res
    res["n_days"] = int(sub.index.nunique())
    res["n_stocks"] = int(sub["ticker"].nunique())
    if n < min_signals:
        return res

    wins = (sub["fwd_cc"] > 0).to_numpy()
    n_pos = int(wins.sum())
    res["hit_rate"] = n_pos / n * 100
    res["avg_return"] = float(sub["fwd_cc"].mean())
    if cluster:
        lo, hi = cluster_mean_ci(wins.astype(float), sub.index.to_numpy())
        res["ci_lo"], res["ci_hi"] = lo * 100, hi * 100
        rlo, rhi = cluster_mean_ci(sub["fwd_cc"].to_numpy(), sub.index.to_numpy())
        res["ret_ci_lo"], res["ret_ci_hi"] = rlo, rhi
    else:
        res["ci_lo"], res["ci_hi"] = wilson_interval(n_pos, n)

    t = sub["fwd_trig"].dropna() - 2 * cost_pct
    res["trig_n"] = len(t)
    if len(t) >= min_signals:
        res["trig_hit_rate"] = float((t > 0).mean() * 100)
        res["trig_avg_return"] = float(t.mean())
    return res


def per_ticker_summary(pool: pd.DataFrame, threshold: int, min_signals: int = MIN_BACKTEST_SIGNALS,
                       cost_pct: float = DEFAULT_COST_PCT) -> pd.DataFrame:
    """Her hisse için (puan >= eşik) tarihsel özet — vektörize groupby."""
    cols = ["n", "hit_rate", "avg_return", "ci_lo", "ci_hi", "trig_n", "trig_hit_rate", "trig_avg_return"]
    if pool.empty:
        return pd.DataFrame(columns=cols)
    sig = pool[pool["score"] >= threshold]
    if sig.empty:
        return pd.DataFrame(columns=cols)
    g = sig.assign(win=(sig["fwd_cc"] > 0)).groupby("ticker")
    out = pd.DataFrame({"n": g.size(), "wins": g["win"].sum(), "avg_return": g["fwd_cc"].mean()})
    out["hit_rate"] = out["wins"] / out["n"] * 100
    lo, hi = wilson_interval_vec(out["wins"], out["n"])
    out["ci_lo"], out["ci_hi"] = lo, hi

    t = sig.dropna(subset=["fwd_trig"]).copy()
    t["net"] = t["fwd_trig"] - 2 * cost_pct
    gt = t.assign(twin=(t["net"] > 0)).groupby("ticker")
    out["trig_n"] = gt.size().reindex(out.index).fillna(0).astype(int)
    out["trig_hit_rate"] = (gt["twin"].mean() * 100).reindex(out.index)
    out["trig_avg_return"] = gt["net"].mean().reindex(out.index)

    weak = out["n"] < min_signals
    for c in ("hit_rate", "avg_return", "ci_lo", "ci_hi"):
        out.loc[weak, c] = np.nan
    weak_t = out["trig_n"] < min_signals
    for c in ("trig_hit_rate", "trig_avg_return"):
        out.loc[weak_t, c] = np.nan
    return out[cols]


def aggregate_summary(pool: pd.DataFrame, threshold: int, min_signals: int = MIN_AGGREGATE_BACKTEST_SIGNALS,
                      cost_pct: float = DEFAULT_COST_PCT) -> dict:
    """Tüm evren: sinyal (puan >= eşik) + 'baz' (tüm hisse-günleri) + fark (lift).
    Baz olmadan %55 gibi bir başarı oranı anlamsızdır: piyasa zaten yükselen bir dönemde
    rastgele bir günün ertesi gün pozitif kapanma oranı da yüksek olabilir."""
    if pool.empty:
        return {"signal": _empty_summary(), "base": _empty_summary(), "lift_hit": np.nan, "lift_return": np.nan}
    signal = summarize_signals(pool[pool["score"] >= threshold], min_signals, cost_pct, cluster=True)
    base = summarize_signals(pool, min_signals, cost_pct, cluster=True)
    lift_hit = signal["hit_rate"] - base["hit_rate"] if pd.notna(signal["hit_rate"]) and pd.notna(base["hit_rate"]) else np.nan
    lift_ret = signal["avg_return"] - base["avg_return"] if pd.notna(signal["avg_return"]) and pd.notna(base["avg_return"]) else np.nan
    return {"signal": signal, "base": base, "lift_hit": lift_hit, "lift_return": lift_ret}


def score_table(pool: pd.DataFrame, max_score: int = MAX_POSITIVE_CRITERIA,
                min_signals: int = MIN_AGGREGATE_BACKTEST_SIGNALS, cost_pct: float = DEFAULT_COST_PCT) -> pd.DataFrame:
    """Puan basamağı başına ertesi gün sonuçları. Puanlama gerçekten ayırt edici ise
    başarı oranı/getiri puanla birlikte (kabaca) artmalıdır — bunu kontrol etmenin tek yolu budur."""
    rows = []
    for s in range(0, max_score + 1):
        sub = pool[pool["score"] == s] if not pool.empty else pool
        r = summarize_signals(sub, min_signals, cost_pct, cluster=False)
        rows.append({
            "Puan": s, "Sinyal-gün": r["n"],
            "Ertesi gün pozitif %": r["hit_rate"], "Ort. getiri % (brüt)": r["avg_return"],
            "Tetik işlem sayısı": r["trig_n"], "Tetik ort. getiri % (net)": r["trig_avg_return"],
        })
    return pd.DataFrame(rows)
