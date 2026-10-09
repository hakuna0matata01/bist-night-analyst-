"""
BIST Akşam Taraması - Faz 1
-----------------------------------
Piyasa kapandıktan sonra (örn. 20:00) çalıştırılarak, ertesi gün yükseliş
potansiyeli yüksek görünen BIST hisselerini teknik kriterlere göre puanlayan
ve sıralayan bir tarama aracı.

Kriterler (her biri 1 puan, toplam 5):
  1. Trend     : Fiyat > EMA20 ve Fiyat > EMA50
  2. Momentum  : RSI(14) 40-65 bandında ve yükseliyor
  3. MACD      : MACD çizgisi sinyalin üzerinde ve histogram güçleniyor
  4. Hacim     : Son günün hacmi 20 günlük ortalamanın üzerinde + fiyat yukarı
  5. Seviye    : Direnç kırılımı / direnç yakınlığı ya da sıkışma sonrası

NOT: Bu araç yatırım tavsiyesi değildir, sadece bir tarama/ön-eleme aracıdır.
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
# Sayfa ayarları
# ----------------------------------------------------------------------------
st.set_page_config(page_title="BIST Akşam Taraması - Faz 1", layout="wide", page_icon="📈")

# Küçük, hızlı denemeler için örnek alt küme (tam liste yüklenemezse yedek olarak da kullanılır)
SAMPLE_TICKERS = [
    "AKBNK", "ALARK", "ASELS", "ASTOR", "BIMAS", "EKGYO", "ENKAI", "EREGL",
    "FROTO", "GARAN", "GUBRF", "HALKB", "ISCTR", "KCHOL", "KONTR", "KOZAL",
    "KRDMD", "ODAS", "OYAKC", "PETKM", "PGSUS", "SAHOL", "SASA", "SISE",
    "TAVHL", "TCELL", "THYAO", "TOASO", "TUPRS", "VAKBN", "YKBNK",
]

TICKERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bist_tickers.txt")

MIN_BARS_REQUIRED = 60  # EMA50 ve sıkışma analizi için gereken minimum gün sayısı
DOWNLOAD_CHUNK_SIZE = 100  # Toplu indirmede tek seferde istenecek hisse sayısı


@st.cache_data(show_spinner=False)
def load_full_bist_list() -> list:
    """data/bist_tickers.txt içindeki tüm BIST hisse kodlarını okur."""
    try:
        with open(TICKERS_FILE, "r", encoding="utf-8") as f:
            codes = [line.strip().upper() for line in f if line.strip()]
        return sorted(set(codes))
    except FileNotFoundError:
        return SAMPLE_TICKERS


# ----------------------------------------------------------------------------
# Veri çekme
# ----------------------------------------------------------------------------
def _chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


@st.cache_data(ttl=7200, show_spinner=False)
def fetch_batch(tickers: tuple, period: str) -> dict:
    """Birden çok hisseyi chunk'lar halinde ve paralel (threads=True) olarak indirir.

    634 hisseyi tek tek sırayla indirmek yerine, yfinance'in çoklu-sembol
    indirme mekanizmasını kullanarak isteği parçalar halinde eş zamanlı
    yapar. Dönüş: {ticker: DataFrame}.
    """
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


def normalize_ticker(raw: str) -> str:
    raw = raw.strip().upper()
    if not raw:
        return raw
    if not raw.endswith(".IS"):
        raw += ".IS"
    return raw


# ----------------------------------------------------------------------------
# Teknik göstergeler
# ----------------------------------------------------------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bandwidth(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return (upper - lower) / sma


# ----------------------------------------------------------------------------
# Faz 1 puanlama mantığı
# ----------------------------------------------------------------------------
def analyze_stock(df: pd.DataFrame):
    close = df["Close"]
    volume = df["Volume"]

    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    rsi14 = rsi(close, 14)
    macd_line, signal_line, hist = macd(close)
    bw = bollinger_bandwidth(close)

    n = len(close)
    idx = n - 1

    # 1. Trend filtresi
    trend_ok = bool(close.iloc[idx] > ema20.iloc[idx] and close.iloc[idx] > ema50.iloc[idx])

    # 2. RSI momentum (40-65 bandı ve yükseliyor)
    rsi_val = float(rsi14.iloc[idx])
    rsi_prev = float(rsi14.iloc[idx - 3]) if idx >= 3 else rsi_val
    rsi_ok = bool(40 <= rsi_val <= 65 and rsi_val > rsi_prev)

    # 3. MACD (sinyal üzerinde, histogram güçleniyor)
    macd_ok = bool(macd_line.iloc[idx] > signal_line.iloc[idx] and hist.iloc[idx] > hist.iloc[idx - 1])

    # 4. Hacim onayı (bugünü hariç tutan 20 günlük ortalamaya göre)
    avg_vol20 = float(volume.iloc[max(0, idx - 20):idx].mean())
    vol_ok = bool(avg_vol20 > 0 and volume.iloc[idx] > avg_vol20 and close.iloc[idx] > close.iloc[idx - 1])

    # 5. Direnç kırılımı / yakınlığı ya da sıkışma sonrası
    recent_high = float(close.iloc[max(0, idx - 20):idx].max()) if idx > 0 else float(close.iloc[idx])
    breakout = bool(close.iloc[idx] >= recent_high)
    near_resistance = bool(close.iloc[idx] >= recent_high * 0.98)
    bw_window = bw.iloc[max(0, idx - 60):idx].dropna()
    squeeze = bool(len(bw_window) >= 30 and bw.iloc[idx] <= bw_window.quantile(0.2))
    level_ok = bool(breakout or near_resistance or squeeze)

    details = {
        f"Trend: Fiyat > EMA20 ({ema20.iloc[idx]:.2f}) & EMA50 ({ema50.iloc[idx]:.2f})": trend_ok,
        f"RSI(14)={rsi_val:.1f} → 40-65 bandında ve yükseliyor": rsi_ok,
        "MACD sinyal üzerinde ve histogram güçleniyor": macd_ok,
        f"Hacim ort.(20g) üzeri (oran: {volume.iloc[idx]/avg_vol20:.2f}x) + fiyat yukarı": vol_ok,
        "Direnç kırılımı / yakınlığı ya da sıkışma sonrası": level_ok,
    }

    score = sum(details.values())

    metrics = {
        "Kapanış": float(close.iloc[idx]),
        "Değişim %": float((close.iloc[idx] / close.iloc[idx - 1] - 1) * 100) if idx > 0 else 0.0,
        "RSI(14)": rsi_val,
        "EMA20": float(ema20.iloc[idx]),
        "EMA50": float(ema50.iloc[idx]),
        "Hacim Oranı": float(volume.iloc[idx] / avg_vol20) if avg_vol20 > 0 else np.nan,
        "Tetik Seviyesi (20g Yüksek)": recent_high,
    }

    return score, details, metrics


def score_color(val) -> str:
    """'Puan' sütunu için matplotlib gerektirmeyen, hafif bir arka plan rengi üretir.

    pandas Styler.background_gradient() dahili olarak matplotlib ister; bu da
    Streamlit Cloud gibi ortamlarda ekstra, ağır bir bağımlılık + kurulum
    süresi demek. Bunun yerine puanı (0-5) yeşilin opaklığına eşleyen basit
    bir CSS üretiyoruz.
    """
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    v = max(0.0, min(5.0, v))
    alpha = 0.12 + (v / 5) * 0.55
    return f"background-color: rgba(22, 163, 74, {alpha:.2f})"


# ----------------------------------------------------------------------------
# Grafik
# ----------------------------------------------------------------------------
def build_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    close = df["Close"]
    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    rsi14 = rsi(close, 14)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.2, 0.25],
        vertical_spacing=0.04, subplot_titles=(ticker, "Hacim", "RSI(14)"),
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Fiyat", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema20, name="EMA20", line=dict(width=1.3, color="#3b82f6")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema50, name="EMA50", line=dict(width=1.3, color="#f97316")), row=1, col=1)

    vol_colors = np.where(df["Close"] >= df["Open"], "#16a34a", "#dc2626")
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors, name="Hacim", showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=rsi14, name="RSI(14)", line=dict(width=1.3, color="#8b5cf6")), row=3, col=1)
    fig.add_hline(y=65, line_dash="dot", line_color="gray", row=3, col=1)
    fig.add_hline(y=40, line_dash="dot", line_color="gray", row=3, col=1)

    fig.update_layout(height=650, margin=dict(l=10, r=10, t=40, b=10), xaxis_rangeslider_visible=False)
    return fig


# ----------------------------------------------------------------------------
# Arayüz
# ----------------------------------------------------------------------------
st.title("📈 BIST Akşam Taraması — Faz 1")
st.caption(
    "Piyasa kapandıktan sonra çalıştırılır. Amaç: ertesi gün yükseliş potansiyeli yüksek "
    "görünen hisselerden kısa bir izleme listesi (watchlist) çıkarmak."
)
st.info(
    "⚠️ Bu araç yatırım tavsiyesi değildir; sadece teknik kriterlere dayalı bir ön-eleme/tarama "
    "aracıdır. Nihai kararı kendi analizinizle birlikte verin.",
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
            f"{len(tickers)} hisse taranacak. `data/bist_tickers.txt` dosyasından okunur — "
            "listeyi güncellemek için o dosyayı değiştirin."
        )
        st.warning(
            "Tüm liste taraması veri sağlayıcı hız sınırlarına bağlı olarak birkaç dakika "
            "sürebilir. Sonuç 2 saat boyunca önbellekte tutulur.",
            icon="⏳",
        )

    period = st.selectbox("Geçmiş veri aralığı", ["6mo", "1y", "2y"], index=0)
    min_score = st.slider("Minimum puan filtresi (izleme listesi için)", 0, 5, 3)

    run = st.button("🔍 Taramayı Başlat", type="primary", width="stretch")

if "scan" not in st.session_state:
    st.session_state.scan = None

if run:
    rows = []
    details_map = {}
    failed = []

    with st.spinner(f"{len(tickers)} hisse için veri indiriliyor... (büyük listelerde birkaç dakika sürebilir)"):
        batch = fetch_batch(tuple(tickers), period)

    progress = st.progress(0, text="Analiz ediliyor...")
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        df = batch.get(ticker)
        if df is None or len(df) < MIN_BARS_REQUIRED:
            failed.append((ticker, "Veri yok / yetersiz"))
        else:
            try:
                score, details, metrics = analyze_stock(df)
                row = {"Hisse": ticker.replace(".IS", ""), "Puan": score, **metrics}
                rows.append(row)
                details_map[ticker] = (details, df)
            except Exception as e:
                failed.append((ticker, str(e)))

        # Her hissede değil, ~%2'lik adımlarla güncelle (634 hissede UI'ı yormamak için)
        if (i + 1) % max(1, total // 50) == 0 or (i + 1) == total:
            progress.progress((i + 1) / total, text=f"Analiz ediliyor... ({i+1}/{total})")

    progress.empty()

    if rows:
        results_df = pd.DataFrame(rows).sort_values("Puan", ascending=False).reset_index(drop=True)
    else:
        results_df = pd.DataFrame()

    st.session_state.scan = {
        "results": results_df,
        "details": details_map,
        "failed": failed,
        "timestamp": datetime.now().strftime("%d.%m.%Y %H:%M"),
    }

scan = st.session_state.scan

if scan is None:
    st.write("👈 Soldan hisse listenizi ve ayarlarınızı seçip **Taramayı Başlat**'a basın.")
else:
    results_df = scan["results"]
    st.caption(f"Son tarama: {scan['timestamp']}")

    if results_df.empty:
        st.warning("Hiçbir hisse için analiz tamamlanamadı. Hisse kodlarını ve bağlantıyı kontrol edin.")
    else:
        watchlist = results_df[results_df["Puan"] >= min_score]

        tab1, tab2 = st.tabs(["📋 İzleme Listesi", "📊 Tüm Sonuçlar"])

        with tab1:
            st.subheader(f"Puanı ≥ {min_score} olan hisseler ({len(watchlist)})")
            if watchlist.empty:
                st.write("Bu eşiği geçen hisse yok. Eşiği düşürmeyi deneyin.")
            else:
                st.dataframe(
                    watchlist.style.format({
                        "Kapanış": "{:.2f}", "Değişim %": "{:+.2f}", "RSI(14)": "{:.1f}",
                        "EMA20": "{:.2f}", "EMA50": "{:.2f}", "Hacim Oranı": "{:.2f}x",
                        "Tetik Seviyesi (20g Yüksek)": "{:.2f}",
                    }).map(score_color, subset=["Puan"]),
                    width="stretch", hide_index=True,
                )
                csv = watchlist.to_csv(index=False).encode("utf-8-sig")
                st.download_button("⬇️ İzleme listesini CSV olarak indir", csv, "izleme_listesi.csv", "text/csv")

        with tab2:
            st.dataframe(
                results_df.style.format({
                    "Kapanış": "{:.2f}", "Değişim %": "{:+.2f}", "RSI(14)": "{:.1f}",
                    "EMA20": "{:.2f}", "EMA50": "{:.2f}", "Hacim Oranı": "{:.2f}x",
                    "Tetik Seviyesi (20g Yüksek)": "{:.2f}",
                }),
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
            details, df = scan["details"][chosen_ticker]
            cols = st.columns(5)
            for col, (label, ok) in zip(cols, details.items()):
                col.metric(label.split(":")[0].split("(")[0].strip()[:14], "✅" if ok else "❌")
            with st.expander("Kriter detayları"):
                for label, ok in details.items():
                    st.write(("✅ " if ok else "❌ ") + label)
            st.plotly_chart(build_chart(df, chosen), width="stretch")
