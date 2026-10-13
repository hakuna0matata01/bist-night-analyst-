"""
BIST Akşam Taraması - Faz 1 (v4)
--------------------------------
Piyasa kapandıktan sonra (örn. 20:00) çalıştırılıp ertesi gün için izleme listesi
çıkaran Streamlit arayüzü. Tüm hesaplama mantığı `engine.py` içindedir (test edilebilir);
bu dosya yalnızca veri indirme + sunumdan sorumludur.

Puanlama özeti: 7 pozitif kriter (+1), 2 risk cezası (-1). Azami puan 7'dir;
BIST100 endeksi alınamazsa Göreli Güç hesaplanamaz ve azami puan 6'ya düşer.
Kural ayrıntıları için engine.py başlığına ve README'ye bakın.

NOT: Yatırım tavsiyesi değildir; yalnızca bir tarama/ön-eleme aracıdır.
"""

import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from engine import (
    APP_VERSION, DEFAULT_COST_PCT, DEFAULT_TRIGGER_BUFFER_PCT, MAX_POSITIVE_CRITERIA,
    MIN_AGGREGATE_BACKTEST_SIGNALS, MIN_BACKTEST_SIGNALS, MIN_BARS_REQUIRED,
    aggregate_summary, build_indicator_set, build_pool, drop_incomplete_last_bar,
    evaluate_today, extract_ticker_frame, index_usable, istanbul_now,
    normalize_index_series, per_ticker_summary, pool_frame, prepare_frame,
    score_table, staleness_reason,
)

# ----------------------------------------------------------------------------
# Sayfa ayarları ve sabitler
# ----------------------------------------------------------------------------
st.set_page_config(page_title=f"BIST Akşam Taraması - Faz 1 ({APP_VERSION})", layout="wide", page_icon="📈")

SAMPLE_TICKERS = [
    "AKBNK", "ALARK", "ASELS", "ASTOR", "BIMAS", "EKGYO", "ENKAI", "EREGL",
    "FROTO", "GARAN", "GUBRF", "HALKB", "ISCTR", "KCHOL", "KONTR", "KOZAL",
    "KRDMD", "ODAS", "OYAKC", "PETKM", "PGSUS", "SAHOL", "SASA", "SISE",
    "TAVHL", "TCELL", "THYAO", "TOASO", "TUPRS", "VAKBN", "YKBNK",
]

TICKERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bist_tickers.txt")
INDEX_TICKER = "XU100.IS"   # BIST 100 endeksi (piyasa rejimi / göreli güç referansı)
DOWNLOAD_CHUNK_SIZE = 100   # Toplu indirmede tek seferde istenecek hisse sayısı
FETCH_RETRIES = 3           # Bir grup/endeks indirmesi başarısız olursa deneme sayısı
FETCH_TIMEOUT = 30          # yfinance istek zaman aşımı (sn)

TIEBREAK_OPTIONS = {
    "Göreli güç (20g)": ("Gör. Güç 20g %", False),
    "Ortalama ciro": ("Ort. Ciro (M TL)", False),
    "Hacim oranı": ("Hacim Oranı", False),
    "Tetiğe yakınlık": ("Tetiğe Uzaklık %", True),
}


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
    return raw if raw.endswith(".IS") else raw + ".IS"


def unique_in_order(items):
    seen, out = set(), []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ----------------------------------------------------------------------------
# Veri çekme (yeniden deneme + başarısız indirmeyi ÖNBELLEĞE ALMAMA)
# ----------------------------------------------------------------------------
def _chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _download(tickers: list, period: str):
    return yf.download(
        tickers=tickers, period=period, interval="1d", group_by="ticker",
        threads=True, progress=False, auto_adjust=True, timeout=FETCH_TIMEOUT,
    )


@st.cache_data(ttl=7200, show_spinner=False)
def fetch_chunk(tickers: tuple, period: str) -> dict:
    """Bir hisse grubunu indirir. Tamamen başarısızsa İSTİSNA fırlatır — böylece
    Streamlit boş/hatalı sonucu 2 saat boyunca önbelleğe almaz (eski sürümde
    geçici bir Yahoo hatası tüm akşamı bozuyordu). Kısmi başarıda eksik hisseler
    bir kez daha denenir."""
    last_err = "boş yanıt"
    for attempt in range(FETCH_RETRIES):
        try:
            raw = _download(list(tickers), period)
            frames = {}
            for t in tickers:
                f = extract_ticker_frame(raw, t, len(tickers))
                if f is not None and not f.empty:
                    frames[t] = f
            if frames:
                missing = [t for t in tickers if t not in frames]
                if missing and len(missing) < len(tickers):
                    try:
                        raw2 = _download(missing, period)
                        for t in missing:
                            f = extract_ticker_frame(raw2, t, len(missing))
                            if f is not None and not f.empty:
                                frames[t] = f
                    except Exception:
                        pass
                return frames
        except Exception as e:  # ağ/ratelimit/parse
            last_err = str(e)
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(last_err)


@st.cache_data(ttl=7200, show_spinner=False)
def fetch_index(period: str) -> pd.DataFrame:
    """BIST100 endeksini (tz'siz, günlük) indirir; başarısızsa istisna (önbelleğe alınmaz)."""
    last_err = "boş yanıt"
    for attempt in range(FETCH_RETRIES):
        try:
            df = yf.download(INDEX_TICKER, period=period, interval="1d", progress=False,
                             auto_adjust=True, timeout=FETCH_TIMEOUT)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            s = normalize_index_series(df["Close"])
            if len(s) > 0:
                return s.to_frame("Close")
        except Exception as e:
            last_err = str(e)
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(last_err)


# ----------------------------------------------------------------------------
# Görsel yardımcılar
# ----------------------------------------------------------------------------
def score_color(val) -> str:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    v = max(0.0, min(MAX_POSITIVE_CRITERIA, v))
    alpha = 0.10 + (v / MAX_POSITIVE_CRITERIA) * 0.55
    return f"background-color: rgba(22, 163, 74, {alpha:.2f})"


def build_chart(df: pd.DataFrame, ind: dict, ticker: str) -> go.Figure:
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, row_heights=[0.40, 0.12, 0.16, 0.16, 0.16],
        vertical_spacing=0.03, subplot_titles=(ticker, "Hacim", "RSI(14)", "MACD", "ADX(14) / +DI / −DI"),
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
    for lvl in (40, 65, 80):
        fig.add_hline(y=lvl, line_dash="dot", line_color="gray", row=3, col=1)

    hist = ind["macd_hist"]
    fig.add_trace(go.Bar(x=df.index, y=hist, name="Histogram",
                         marker_color=np.where(hist.fillna(0) >= 0, "#16a34a", "#dc2626"), showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ind["macd_line"], name="MACD", line=dict(width=1.2, color="#2563eb")), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ind["signal_line"], name="Sinyal", line=dict(width=1.2, color="#f59e0b")), row=4, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=ind["adx14"], name="ADX(14)", line=dict(width=1.5, color="#0f766e")), row=5, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ind["plus_di"], name="+DI", line=dict(width=1, color="#16a34a", dash="dot")), row=5, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ind["minus_di"], name="−DI", line=dict(width=1, color="#dc2626", dash="dot")), row=5, col=1)

    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])  # hafta sonu boşlukları
    fig.update_layout(height=920, margin=dict(l=10, r=10, t=40, b=10), xaxis_rangeslider_visible=False)
    return fig


# ----------------------------------------------------------------------------
# Arayüz
# ----------------------------------------------------------------------------
st.title(f"📈 BIST Akşam Taraması — Faz 1 ({APP_VERSION})")
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
    st.header("Tarama ayarları")
    st.caption("Bu bölümdeki değişiklikler **yeniden tarama** gerektirir.")

    source = st.radio(
        "Hisse listesi kaynağı",
        [f"Tüm BIST Hisseleri ({len(full_list)})", "Örnek alt küme (BIST30)", "Kendi listem"],
    )
    if source == "Kendi listem":
        raw_input = st.text_area(
            "Hisse kodlarını virgülle ayırarak girin (.IS gerekmez)",
            value="THYAO, ASELS, SASA, FROTO", height=100,
        )
        tickers = unique_in_order([normalize_ticker(t) for t in raw_input.split(",")])
    elif source.startswith("Örnek"):
        tickers = [normalize_ticker(t) for t in SAMPLE_TICKERS]
        st.caption(f"{len(tickers)} hisse — hızlı deneme için küçük bir alt küme.")
    else:
        tickers = [normalize_ticker(t) for t in full_list]
        st.caption(f"{len(tickers)} hisse taranacak. `data/bist_tickers.txt` dosyasından okunur.")
        st.warning("Tüm liste taraması birkaç dakika sürebilir. Sonuçlar 2 saat önbellekte tutulur.", icon="⏳")

    period = st.selectbox("Geçmiş veri aralığı", ["6mo", "1y", "2y", "5y"], index=1)
    st.caption("Daha uzun aralık, tarihsel doğrulama için daha büyük örneklem sağlar (5y daha yavaştır).")

    adx_threshold = st.slider("Minimum ADX(14) — trend gücü eşiği", 10, 40, 20)
    trigger_buffer_pct = st.number_input(
        "Giriş tetik tamponu (%) — referans direncin üzerine eklenir",
        min_value=0.0, max_value=5.0, value=DEFAULT_TRIGGER_BUFFER_PCT, step=0.1,
    )
    st.caption(
        "Yarın izlenecek fiyat = (dünkü direnç ya da bugünün kapanışından büyük olanı) "
        "× (1 + tampon). Sabit kuruş yerine yüzde kullanılır."
    )

    run = st.button("🔍 Taramayı Başlat", type="primary", width="stretch")

    st.divider()
    st.header("Görüntüleme filtreleri")
    st.caption("Bunlar **anında** uygulanır; yeniden tarama gerekmez.")
    min_score = st.slider(f"Minimum puan filtresi (0-{MAX_POSITIVE_CRITERIA})", 0, MAX_POSITIVE_CRITERIA, 4)
    st.caption(
        f"Azami puan **{MAX_POSITIVE_CRITERIA}**'dir. BIST100 endeks verisi alınamazsa Göreli Güç "
        "hesaplanamaz ve o taramada azami puan 6'ya düşer (üstte uyarı çıkar)."
    )
    min_turnover_m = st.number_input("Minimum ort. günlük ciro (Milyon TL, 20g)", min_value=0.0, value=20.0, step=5.0)
    exclude_illiquid = st.checkbox("Düşük likiditeli hisseleri izleme listesinden hariç tut", value=True)
    tiebreak_label = st.selectbox("Eşit puanda sıralama ölçütü", list(TIEBREAK_OPTIONS))
    cost_pct = st.number_input(
        "İşlem maliyeti, tek yön (%)", min_value=0.0, max_value=1.0, value=DEFAULT_COST_PCT, step=0.05,
        help="Tetik bazlı backtest'te alış ve satış için ayrı ayrı düşülür (komisyon + kayma tahmini).",
    )

if "scan" not in st.session_state:
    st.session_state.scan = None

# ----------------------------------------------------------------------------
# Tarama
# ----------------------------------------------------------------------------
if run:
    now = istanbul_now()

    index_close, index_error = None, None
    with st.spinner("BIST100 endeks verisi alınıyor..."):
        try:
            idx_df = fetch_index(period)
            idx_df, idx_dropped = drop_incomplete_last_bar(idx_df, now)
            index_close = idx_df["Close"]
        except Exception as e:
            index_error = str(e)

    chunks = list(_chunked(tickers, DOWNLOAD_CHUNK_SIZE))
    data, failed = {}, []
    dl_bar = st.progress(0, text="Veri indiriliyor...")
    for ci, chunk in enumerate(chunks):
        try:
            data.update(fetch_chunk(tuple(chunk), period))
        except Exception as e:
            failed.extend((t, f"İndirme hatası: {e}") for t in chunk)
        dl_bar.progress((ci + 1) / len(chunks), text=f"Veri indiriliyor... ({ci + 1}/{len(chunks)})")
    dl_bar.empty()

    # Seans bitmeden çalıştırıldıysa bugünkü yarım mumu çıkar
    dropped_incomplete = False
    for t in list(data):
        data[t], dropped = drop_incomplete_last_bar(data[t], now)
        dropped_incomplete |= dropped
        if data[t].empty:
            del data[t]

    ref_date = max((df.index[-1] for df in data.values()), default=None)
    index_ok = index_usable(index_close, ref_date)

    rows, stored, pool_frames = [], {}, []
    quality_issues, trimmed_notes = [], []
    failed_set = {t for t, _ in failed}
    total = len(tickers)
    progress = st.progress(0, text="Analiz ediliyor...")

    for i, ticker in enumerate(tickers):
        if ticker not in failed_set:
            df = data.get(ticker)
            if df is None or len(df) < MIN_BARS_REQUIRED:
                failed.append((ticker, "Veri yok / yetersiz"))
            else:
                stale = staleness_reason(df, ref_date) if ref_date is not None else None
                df2, note, issue = prepare_frame(df)
                if stale:
                    failed.append((ticker, stale))
                elif issue:
                    quality_issues.append((ticker, issue))
                elif df2 is None or len(df2) < MIN_BARS_REQUIRED:
                    failed.append((ticker, "Veri yok / yetersiz"))
                else:
                    try:
                        ind = build_indicator_set(df2, index_close, adx_threshold, trigger_buffer_pct)
                        score, details, risk, metrics = evaluate_today(ind)
                        code = ticker.replace(".IS", "")
                        rows.append({"Hisse": code, "Puan": score, **metrics,
                                     "Son Veri": df2.index[-1].date().isoformat()})
                        stored[ticker] = df2
                        pool_frames.append(pool_frame(code, ind))
                        if note:
                            trimmed_notes.append((ticker, note))
                    except Exception as e:
                        failed.append((ticker, f"Hesaplama hatası: {e}"))
        if (i + 1) % max(1, total // 50) == 0 or (i + 1) == total:
            progress.progress((i + 1) / total, text=f"Analiz ediliyor... ({i + 1}/{total})")
    progress.empty()

    st.session_state.scan = {
        "results": pd.DataFrame(rows),
        "frames": stored,
        "pool": build_pool(pool_frames),
        "failed": failed,
        "quality_issues": quality_issues,
        "trimmed_notes": trimmed_notes,
        "index_close": index_close,
        "index_available": index_ok,
        "index_error": index_error,
        "dropped_incomplete": dropped_incomplete,
        "adx_threshold": adx_threshold,
        "trigger_buffer_pct": trigger_buffer_pct,
        "ref_date": ref_date.date().isoformat() if ref_date is not None else "—",
        "timestamp": now.strftime("%d.%m.%Y %H:%M"),
        "file_stamp": now.strftime("%Y%m%d_%H%M"),
    }

scan = st.session_state.scan

# ----------------------------------------------------------------------------
# Sonuçlar
# ----------------------------------------------------------------------------
if scan is None:
    st.write("👈 Soldan hisse listenizi ve ayarlarınızı seçip **Taramayı Başlat**'a basın.")
else:
    st.caption(f"Son tarama: {scan['timestamp']} · Verinin son işlem günü: {scan['ref_date']}")

    if scan.get("dropped_incomplete"):
        st.info(
            "⏱️ Tarama seans bitmeden (18:15 TSİ öncesi) çalıştırıldığı için bugünün tamamlanmamış "
            "mumu hesaplara dahil edilmedi; sonuçlar bir önceki kapanışa göre hesaplandı.",
        )

    # --- Piyasa rejimi / endeks durumu ---
    index_available = scan["index_available"]
    idx_close = scan.get("index_close")
    if not index_available:
        why = f" Hata: {scan['index_error']}." if scan.get("index_error") else ""
        st.error(
            "⚠️ BIST100 endeks verisi alınamadı ya da güncel değil! 'Göreli Güç' kriteri bu taramada "
            f"hesaplanamadı — bu taramadaki azami puan 7 değil **{MAX_POSITIVE_CRITERIA - 1}** olarak "
            f"değerlendirilmelidir.{why} Taramayı tekrar deneyin.",
            icon="⚠️",
        )
    elif len(idx_close) >= 50:
        idx_ema50 = idx_close.ewm(span=50, adjust=False).mean()
        idx_up = bool(idx_close.iloc[-1] > idx_ema50.iloc[-1])
        idx_chg = float((idx_close.iloc[-1] / idx_close.iloc[-2] - 1) * 100) if len(idx_close) > 1 else 0.0
        if idx_up:
            st.success(f"📊 BIST100 endeksi EMA50 üzerinde (genel eğilim yukarı yönlü). Son kapanış değişimi: {idx_chg:+.2f}%")
        else:
            st.warning(
                "📊 BIST100 endeksi EMA50 altında (genel eğilim zayıf/aşağı yönlü) — bireysel al "
                f"sinyallerinin güvenilirliği böyle ortamlarda genelde düşer. Son kapanış değişimi: {idx_chg:+.2f}%"
            )
    else:
        st.caption(
            f"ℹ️ Endeks geçmişi sınırlı ({len(idx_close)} bar): Göreli Güç hesaplandı (azami puan 7), "
            "ancak EMA50 bazlı piyasa rejimi notu için yeterli veri yok."
        )

    results_raw = scan["results"]
    pool = scan["pool"]

    if results_raw.empty:
        st.warning("Hiçbir hisse için analiz tamamlanamadı. Hisse kodlarını ve bağlantıyı kontrol edin.")
    else:
        # --- Canlı filtreler: tarihsel özet + likidite + sıralama ---
        hist = per_ticker_summary(pool, min_score, MIN_BACKTEST_SIGNALS, cost_pct)
        res = results_raw.join(hist, on="Hisse").rename(columns={
            "n": "Sinyal Sayısı", "hit_rate": "Tarihsel Başarı %", "avg_return": "Ort. Sonraki Gün Getiri %",
            "trig_n": "Tetik İşlem Sayısı", "trig_hit_rate": "Tetik Başarı %", "trig_avg_return": "Tetik Net Getiri %",
        })
        for c in ("Sinyal Sayısı", "Tetik İşlem Sayısı"):
            res[c] = res[c].fillna(0).astype(int)
        res["Başarı %95 GA"] = [
            f"{lo:.0f}–{hi:.0f}" if pd.notna(lo) else "—" for lo, hi in zip(res["ci_lo"], res["ci_hi"])
        ]
        res["Likidite OK"] = res["Ort. Ciro (M TL)"].fillna(-1) >= min_turnover_m
        tb_col, tb_asc = TIEBREAK_OPTIONS[tiebreak_label]
        res = res.sort_values(["Puan", tb_col], ascending=[False, tb_asc], na_position="last").reset_index(drop=True)

        # --- Tüm evren bazlı doğrulama ---
        agg = aggregate_summary(pool, min_score, MIN_AGGREGATE_BACKTEST_SIGNALS, cost_pct)
        sig, base = agg["signal"], agg["base"]
        if pd.notna(sig["hit_rate"]):
            tetik_txt = ""
            if pd.notna(sig["trig_avg_return"]):
                tetik_txt = (
                    f" **Gerçekçi (tetik bazlı) senaryo:** {sig['trig_n']} işlem, maliyet sonrası ortalama "
                    f"**{sig['trig_avg_return']:+.2f}%**, pozitif oranı **%{sig['trig_hit_rate']:.0f}**."
                )
            st.info(
                f"📐 **Tüm evren bazlı tarihsel doğrulama** (puan ≥ {min_score}; {sig['n_stocks']} hisse, "
                f"{sig['n_days']} farklı gün): **{sig['n']}** hisse-gün sinyali, ertesi gün pozitif kapanış "
                f"**%{sig['hit_rate']:.0f}** (gün-kümelenmeli %95 GA: {sig['ci_lo']:.0f}–{sig['ci_hi']:.0f}). "
                f"Kıyas için tüm hisse-günlerinde bu oran **%{base['hit_rate']:.0f}** → sinyalin katkısı "
                f"**{agg['lift_hit']:+.1f} puan**; ortalama getiri sinyalde **{sig['avg_return']:+.2f}%**, "
                f"baz **{base['avg_return']:+.2f}%** (fark {agg['lift_return']:+.2f}).{tetik_txt} "
                "Sinyaller aynı günlerde kümelendiği için aralıklar günlere göre hesaplandı; yine de "
                "hisse listesi bugünün hisselerinden oluştuğundan (hayatta kalma yanlılığı) sonuçlar "
                "iyimser olabilir.",
                icon="📐",
            )
        else:
            st.caption(
                f"ℹ️ Tüm evren bazlı doğrulama için yeterli sinyal yok (bulunan: {sig['n']}, gereken minimum: "
                f"{MIN_AGGREGATE_BACKTEST_SIGNALS}). Daha fazla hisse, daha uzun veri aralığı ya da daha düşük puan eşiği deneyin."
            )

        with st.expander("🧪 Puan basamağına göre sonuçlar (puanlama gerçekten ayırt edici mi?)"):
            st.caption(
                "Puanlama işe yarıyorsa, yüksek puanlarda ertesi gün pozitif oranı ve ortalama getiri düşük "
                "puanlardan belirgin biçimde yüksek olmalıdır. Düz ya da ters bir tablo, kuralların "
                "o dönemde ayırt edici olmadığını gösterir."
            )
            stbl = score_table(pool, MAX_POSITIVE_CRITERIA, MIN_AGGREGATE_BACKTEST_SIGNALS, cost_pct)
            st.dataframe(
                stbl.style.format({
                    "Ertesi gün pozitif %": "{:.1f}", "Ort. getiri % (brüt)": "{:+.2f}",
                    "Tetik ort. getiri % (net)": "{:+.2f}",
                }, na_rep="—"),
                width="stretch", hide_index=True,
            )

        if scan.get("quality_issues"):
            with st.expander(f"🚫 Veri kalitesi nedeniyle hariç tutulan {len(scan['quality_issues'])} hisse"):
                st.caption(
                    "Bu hisselerde yfinance'in doğru ayarlamadığı bir bedelsiz/bölünme izi bulundu ve "
                    "sıçramadan sonraki geçmiş analiz için yetersiz kaldı."
                )
                for t, reason in scan["quality_issues"]:
                    st.write(f"- {t}: {reason}")
        if scan.get("trimmed_notes"):
            with st.expander(f"✂️ Geçmişi kırpılarak analiz edilen {len(scan['trimmed_notes'])} hisse"):
                for t, note in scan["trimmed_notes"]:
                    st.write(f"- {t}: {note}")

        base_filter = res["Puan"] >= min_score
        if exclude_illiquid:
            base_filter &= res["Likidite OK"]
        watchlist = res[base_filter]

        display_cols = [
            "Hisse", "Puan", "Ham Puan", "Risk Cezası", "Kapanış", "Değişim %",
            "Giriş Tetik Fiyatı", "Tetiğe Uzaklık %", "Referans Direnç",
            "RSI(14)", "ADX(14)", "Hacim Oranı", "Gör. Güç 20g %", "Ort. Ciro (M TL)",
            "Tarihsel Başarı %", "Başarı %95 GA", "Sinyal Sayısı", "Ort. Sonraki Gün Getiri %",
            "Tetik Başarı %", "Tetik Net Getiri %", "Tetik İşlem Sayısı",
        ]
        fmt = {
            "Kapanış": "{:.2f}", "Değişim %": "{:+.2f}",
            "Giriş Tetik Fiyatı": "{:.2f}", "Tetiğe Uzaklık %": "{:+.2f}", "Referans Direnç": "{:.2f}",
            "RSI(14)": "{:.1f}", "ADX(14)": "{:.1f}", "Hacim Oranı": "{:.2f}x", "Gör. Güç 20g %": "{:+.1f}",
            "Ort. Ciro (M TL)": "{:.1f}", "Tarihsel Başarı %": "{:.0f}",
            "Ort. Sonraki Gün Getiri %": "{:+.2f}", "Tetik Başarı %": "{:.0f}", "Tetik Net Getiri %": "{:+.2f}",
        }

        tab1, tab2 = st.tabs(["📋 İzleme Listesi", "📊 Tüm Sonuçlar"])
        with tab1:
            st.subheader(f"Puanı ≥ {min_score} olan hisseler ({len(watchlist)})")
            if exclude_illiquid:
                st.caption("Düşük likiditeli hisseler bu listeden hariç tutuldu (sol menüden değiştirilebilir).")
            st.caption(
                "'Tarihsel Başarı' = sinyal günü kapanışından ertesi gün kapanışına (brüt). 'Tetik' sütunları ise "
                "ertesi gün giriş tetiği görülürse tetik/açılış fiyatından alıp kapanışta satan gerçekçi senaryodur "
                "(işlem maliyeti düşülmüş)."
            )
            if watchlist.empty:
                st.write("Bu eşiği geçen hisse yok. Eşiği düşürmeyi veya likidite filtresini gevşetmeyi deneyin.")
            else:
                st.dataframe(
                    watchlist[display_cols].style.format(fmt, na_rep="—").map(score_color, subset=["Puan"]),
                    width="stretch", hide_index=True,
                )
                st.download_button(
                    "⬇️ İzleme listesini CSV olarak indir",
                    watchlist[display_cols].to_csv(index=False).encode("utf-8-sig"),
                    f"izleme_listesi_{scan['file_stamp']}.csv", "text/csv",
                )
        with tab2:
            st.dataframe(
                res[display_cols + ["Likidite OK", "Son Veri"]].style.format(fmt, na_rep="—"),
                width="stretch", hide_index=True,
            )
            st.download_button(
                "⬇️ Tüm sonuçları CSV olarak indir",
                res[display_cols + ["Likidite OK", "Son Veri"]].to_csv(index=False).encode("utf-8-sig"),
                f"tum_sonuclar_{scan['file_stamp']}.csv", "text/csv",
            )

        if scan["failed"]:
            with st.expander(f"⚠️ Analiz edilemeyen {len(scan['failed'])} hisse"):
                for t, reason in scan["failed"]:
                    st.write(f"- {t}: {reason}")

        # --- Detay ---
        st.divider()
        st.subheader("Detay İncele")
        chosen = st.selectbox("Bir hisse seçin", res["Hisse"].tolist())
        chosen_ticker = f"{chosen}.IS"
        if chosen_ticker in scan["frames"]:
            df = scan["frames"][chosen_ticker]
            # Detay için göstergeler seçilen hisse özelinde yeniden hesaplanır (bellek tasarrufu).
            ind = build_indicator_set(df, scan["index_close"], scan["adx_threshold"], scan["trigger_buffer_pct"])
            score, details, risk, metrics = evaluate_today(ind)
            row = res[res["Hisse"] == chosen].iloc[0]
            azami = metrics["Azami Puan"]

            st.write(
                f"**Puan: {score} / {azami}** — ham {metrics['Ham Puan']} − risk cezası {metrics['Risk Cezası']}"
            )
            if azami < MAX_POSITIVE_CRITERIA:
                st.caption("ℹ️ Endeks verisi alınamadığı için azami puan bu taramada 7 değil 6.")

            trig_cols = st.columns(3)
            trig_cols[0].metric("Giriş Tetik Fiyatı", f"{row['Giriş Tetik Fiyatı']:.2f} TL")
            trig_cols[1].metric("Referans Direnç", f"{row['Referans Direnç']:.2f} TL")
            trig_cols[2].metric("Tetiğe Uzaklık", f"{row['Tetiğe Uzaklık %']:+.2f}%")
            st.caption(
                f"Yarın fiyat **{row['Giriş Tetik Fiyatı']:.2f} TL** seviyesini yukarı keserse, Faz 2 "
                "kriterleri (yüksek Relative Volume + yön teyidi) sağlanırsa giriş düşünülebilir. "
                "Bu seviye otomatik emir değildir, sadece izlenecek referanstır."
            )

            with st.expander("✅ Kriter ve risk detayları", expanded=True):
                st.markdown("**Pozitif kriterler:**")
                for label, ok in details.items():
                    st.write(("✅ " if ok else "➖ ") + label)
                st.markdown("**Risk bayrakları:**")
                for label, flagged in risk.items():
                    st.write(("🚩 " if flagged else "✅ Yok — ") + label)

            if pd.notna(row["Tarihsel Başarı %"]):
                st.info(
                    f"📈 **Bu hissenin geçmişi** (puan ≥ {min_score}): {int(row['Sinyal Sayısı'])} sinyal, "
                    f"**%{row['Tarihsel Başarı %']:.0f}**'i ertesi gün pozitif kapandı (%95 GA: **{row['Başarı %95 GA']}**), "
                    f"ortalama ertesi gün getirisi **{row['Ort. Sonraki Gün Getiri %']:+.2f}%**. "
                    "Düşük sinyal sayısında bu aralık çok geniştir — tek başına karar vermek için yeterli değildir.",
                )
            else:
                st.caption(
                    f"ℹ️ Tarihsel başarı oranı için yeterli geçmiş sinyal yok (bulunan: {int(row['Sinyal Sayısı'])}, "
                    f"gereken minimum: {MIN_BACKTEST_SIGNALS}). Daha uzun bir veri aralığı deneyin."
                )
            if pd.notna(row["Tetik Net Getiri %"]):
                st.caption(
                    f"🎯 Tetik bazlı senaryo: {int(row['Tetik İşlem Sayısı'])} işlemde pozitif oranı "
                    f"%{row['Tetik Başarı %']:.0f}, maliyet sonrası ortalama {row['Tetik Net Getiri %']:+.2f}%."
                )

            st.plotly_chart(build_chart(df, ind, chosen), width="stretch")
