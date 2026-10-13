"""engine.py için testler. Çalıştırma: `pytest -q` (proje kökünden)."""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine as E  # noqa: E402


# ----------------------------------------------------------------------------
# Yardımcı: sentetik OHLCV
# ----------------------------------------------------------------------------
def synth(seed=0, n=260, drift=0.0015, vol=0.02, end="2026-10-13"):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(r))
    idx = pd.bdate_range(end=end, periods=n)
    high = close * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.008, n)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.004, n))
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])
    volume = rng.lognormal(15, 0.4, n) * (1 + (r > 0.01) * 1.2)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)


# ----------------------------------------------------------------------------
# 1) AZAMİ PUAN: 7 (endeks var) / 6 (endeks yok)
# ----------------------------------------------------------------------------
def test_criteria_count_matches_constant():
    ind = E.build_indicator_set(synth(1), synth(999)["Close"])
    assert len(ind["criteria"]) == E.MAX_POSITIVE_CRITERIA == 7
    assert len(ind["risk_flags"]) == 2


def test_max_score_is_7_with_index_and_6_without():
    scores_with, scores_without = [], []
    for seed in range(120):
        df = synth(seed, drift=0.002 if seed % 2 else 0.0008)
        idx_close = synth(10_000 + seed, drift=0.0003)["Close"]
        a = E.build_indicator_set(df, idx_close)
        b = E.build_indicator_set(df, None)
        assert a["max_score"] == 7 and a["index_available"]
        assert b["max_score"] == 6 and not b["index_available"]
        scores_with.append(a["score_series"].max())
        scores_without.append(b["score_series"].max())
    assert max(scores_with) == 7        # 7'ye ulaşılabiliyor
    assert max(scores_without) == 6     # endeks yokken 6'yı geçemez
    assert min(scores_with) >= 0        # ceza puanı eksiye düşürmez


def test_net_score_equals_raw_minus_penalty_clipped():
    ind = E.build_indicator_set(synth(3), synth(77)["Close"])
    expected = (ind["raw_score_series"] - ind["penalty_series"]).clip(lower=0)
    assert (ind["score_series"] == expected).all()


# ----------------------------------------------------------------------------
# 2) NEDENSELLİK: gelecek veri geçmiş puanı değiştirmemeli (look-ahead yok)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_lookahead_in_scores(seed):
    df = synth(seed, n=300)
    idx_close = synth(500 + seed, n=300)["Close"]
    full = E.build_indicator_set(df, idx_close)["score_series"]
    for k in (150, 200, 250):
        part = E.build_indicator_set(df.iloc[:k], idx_close.iloc[:k])["score_series"]
        assert part.iloc[-1] == full.iloc[k - 1], f"seed={seed}, k={k}"


# ----------------------------------------------------------------------------
# 3) Göstergeler
# ----------------------------------------------------------------------------
def test_rsi_matches_reference_wilder():
    close = synth(5)["Close"]
    period = 14
    d = close.diff().to_numpy()
    gain, loss = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    ag = np.nanmean(gain[1:period + 1])
    al = np.nanmean(loss[1:period + 1])
    ref = {period: 100 - 100 / (1 + ag / al)}
    for i in range(period + 1, len(close)):
        ag = (ag * (period - 1) + gain[i]) / period
        al = (al * (period - 1) + loss[i]) / period
        ref[i] = 100 - 100 / (1 + ag / al)
    out = E.rsi(close, period)
    for i, v in ref.items():
        assert out.iloc[i] == pytest.approx(v, abs=1e-9)
    assert out.iloc[:period].isna().all()   # ısınma dönemi NaN (50 ile doldurulmaz)


def test_rsi_edge_cases():
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert E.rsi(up).iloc[-1] == 100.0
    flat = pd.Series(np.full(60, 10.0))
    assert E.rsi(flat).iloc[-1] == 50.0


def test_wilson_interval_bounds():
    lo, hi = E.wilson_interval(6, 6)
    assert 0 <= lo <= hi <= 100 and hi == pytest.approx(100.0, abs=1e-6)
    lo, hi = E.wilson_interval(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-9) and hi < 40
    assert np.isnan(E.wilson_interval(0, 0)[0])


def test_cluster_ci_is_wider_than_naive_when_clustered():
    rng = np.random.default_rng(0)
    days = np.repeat(np.arange(40), 25)                    # 40 gün x 25 hisse
    day_effect = rng.normal(0, 1, 40)[days]                # günlük ortak şok
    x = (day_effect + rng.normal(0, 0.3, len(days)) > 0).astype(float)
    n_pos = int(x.sum())
    naive_lo, naive_hi = E.wilson_interval(n_pos, len(x))
    c_lo, c_hi = E.cluster_mean_ci(x, days)
    assert (c_hi - c_lo) * 100 > (naive_hi - naive_lo)


# ----------------------------------------------------------------------------
# 4) Veri hijyeni
# ----------------------------------------------------------------------------
def test_tz_aware_stock_and_naive_index_align():
    df = synth(7)
    df_tz = df.copy()
    df_tz.index = df_tz.index.tz_localize("Europe/Istanbul")
    clean = E.clean_ohlcv(df_tz)
    assert clean.index.tz is None
    idx_tz = synth(8)["Close"]
    idx_tz.index = idx_tz.index.tz_localize("Europe/Istanbul")
    ind = E.build_indicator_set(clean, idx_tz)          # eski kod burada TypeError fırlatıyordu
    assert ind["index_available"] and ind["max_score"] == 7


def test_extract_ticker_frame_single_multiindex_and_flat():
    df = synth(9, n=100)
    multi = pd.concat({"THYAO.IS": df}, axis=1)         # (ticker, alan) — yfinance tek hisse biçimi
    assert E.extract_ticker_frame(multi, "THYAO.IS", 1) is not None
    swapped = multi.swaplevel(axis=1)                   # (alan, ticker)
    assert E.extract_ticker_frame(swapped, "THYAO.IS", 1) is not None
    assert E.extract_ticker_frame(df, "THYAO.IS", 1) is not None   # düz sütun
    assert E.extract_ticker_frame(df, "THYAO.IS", 5) is None       # çok hisse + düz sütun = güvenilmez
    two = pd.concat({"A.IS": df, "B.IS": df}, axis=1)
    assert E.extract_ticker_frame(two, "C.IS", 2) is None


def test_prepare_frame_trims_old_jump_and_rejects_recent_one():
    df = synth(11, n=300)
    old = df.copy()
    old.iloc[40:, :4] *= 1.6                            # 40. barda %60 sıçrama (eski)
    out, note, issue = E.prepare_frame(old)
    assert out is not None and note and issue is None and len(out) == 260

    recent = df.copy()
    recent.iloc[290:, :4] *= 1.6                        # çok yakın -> kırpınca yetersiz
    out, note, issue = E.prepare_frame(recent)
    assert out is None and issue


def test_staleness_and_zero_volume():
    df = synth(12, n=120)
    ref = df.index[-1]
    assert E.staleness_reason(df, ref) is None
    assert E.staleness_reason(df.iloc[:-10], ref) is not None
    z = df.copy()
    z.iloc[-1, z.columns.get_loc("Volume")] = 0
    assert "hacim 0" in E.staleness_reason(z, ref).lower()


def test_drop_incomplete_last_bar():
    df = synth(13, n=100, end="2026-10-13")             # son bar = 13 Ekim 2026 (Salı)
    out, dropped = E.drop_incomplete_last_bar(df, datetime(2026, 10, 13, 14, 0))
    assert dropped and len(out) == 99
    out, dropped = E.drop_incomplete_last_bar(df, datetime(2026, 10, 13, 20, 0))
    assert not dropped and len(out) == 100
    out, dropped = E.drop_incomplete_last_bar(df, datetime(2026, 10, 14, 11, 0))
    assert not dropped                                   # son bar dün; bugünkü bar yok


# ----------------------------------------------------------------------------
# 5) Tetik bazlı backtest matematiği (elle hesaplanmış)
# ----------------------------------------------------------------------------
def test_trigger_forward_return_hand_computed():
    idx = pd.bdate_range("2026-01-05", periods=5)
    df = pd.DataFrame({
        "Open":   [100, 101, 106, 100, 100],
        "High":   [101, 103, 108, 101, 100],
        "Low":    [ 99,  99, 104,  99,  99],
        "Close":  [100, 102, 107, 100, 100],
        "Volume": [1] * 5,
    }, index=idx, dtype=float)
    trig = pd.Series([102.0, 105.0, 120.0, 100.5, 100.0], index=idx)
    ret, triggered = E.trigger_forward_return(df, trig)
    # t=0: ertesi High 103 >= 102 -> açılış 101 < 102 => tetikten 102 dolum, kapanış 102 => 0%
    assert ret.iloc[0] == pytest.approx(0.0)
    # t=1: ertesi High 108 >= 105, açılış 106 > 105 (gap) => açılıştan 106, kapanış 107 => +0.943%
    assert ret.iloc[1] == pytest.approx((107 / 106 - 1) * 100)
    # t=2: ertesi High 101 < 120 => tetik görülmedi
    assert pd.isna(ret.iloc[2]) and not triggered.iloc[2]
    # t=3: ertesi High 100 < 100.5 => tetik yok
    assert pd.isna(ret.iloc[3])
    # son bar: ertesi gün yok
    assert pd.isna(ret.iloc[4])


def test_trigger_skips_limit_up_open():
    idx = pd.bdate_range("2026-01-05", periods=2)
    df = pd.DataFrame({"Open": [100, 110], "High": [100, 110], "Low": [100, 110],
                       "Close": [100, 110], "Volume": [1, 1]}, index=idx, dtype=float)
    ret, triggered = E.trigger_forward_return(df, pd.Series([101.0, 101.0], index=idx))
    assert triggered.iloc[0] and pd.isna(ret.iloc[0])   # tavan açılış: işlem yapılamaz


# ----------------------------------------------------------------------------
# 6) Havuz / özetler
# ----------------------------------------------------------------------------
def _pool_from_synth(n_stocks=12, seed0=100):
    idx_close = synth(seed0 + 999)["Close"]
    frames = []
    for s in range(n_stocks):
        ind = E.build_indicator_set(synth(seed0 + s), idx_close)
        frames.append(E.pool_frame(f"T{s}", ind))
    return E.build_pool(frames)


def test_pool_excludes_warmup_and_last_bar():
    df = synth(21)
    ind = E.build_indicator_set(df, None)
    pf = E.pool_frame("X", ind)
    assert pf.index.min() >= df.index[E.WARMUP_BARS]
    assert pf.index.max() < df.index[-1]                 # ertesi günü bilinmeyen son bar yok
    assert pf["fwd_cc"].notna().all()


def test_aggregate_summary_and_score_table_consistency():
    pool = _pool_from_synth()
    assert not pool.empty
    agg = E.aggregate_summary(pool, threshold=3, min_signals=30)
    sig, base = agg["signal"], agg["base"]
    assert base["n"] == len(pool)
    assert sig["n"] == int((pool["score"] >= 3).sum())
    assert 0 <= sig["hit_rate"] <= 100 and sig["ci_lo"] <= sig["hit_rate"] <= sig["ci_hi"]
    assert agg["lift_hit"] == pytest.approx(sig["hit_rate"] - base["hit_rate"])

    tbl = E.score_table(pool, 7, min_signals=30)
    assert list(tbl["Puan"]) == list(range(8))
    assert tbl["Sinyal-gün"].sum() == len(pool)


def test_per_ticker_summary_matches_manual():
    pool = _pool_from_synth()
    out = E.per_ticker_summary(pool, threshold=2, min_signals=5, cost_pct=0.1)
    t = out.dropna(subset=["hit_rate"]).index[0]
    sub = pool[(pool["ticker"] == t) & (pool["score"] >= 2)]
    assert out.loc[t, "n"] == len(sub)
    assert out.loc[t, "hit_rate"] == pytest.approx((sub["fwd_cc"] > 0).mean() * 100)
    assert out.loc[t, "avg_return"] == pytest.approx(sub["fwd_cc"].mean())
    trig = sub["fwd_trig"].dropna() - 0.2
    if len(trig) >= 5:
        assert out.loc[t, "trig_avg_return"] == pytest.approx(trig.mean())


def test_empty_pool_is_safe():
    empty = E.build_pool([])
    assert E.per_ticker_summary(empty, 4).empty
    agg = E.aggregate_summary(empty, 4)
    assert agg["signal"]["n"] == 0
    assert len(E.score_table(empty)) == 8


# ----------------------------------------------------------------------------
# 7) Kural anahtarları (v3 davranışına dönülebilmeli)
# ----------------------------------------------------------------------------
def test_rule_switches_change_only_intended_criteria(monkeypatch):
    df, idx_close = synth(31), synth(32)["Close"]
    new = E.build_indicator_set(df, idx_close)
    monkeypatch.setattr(E, "ADX_REQUIRE_DI_DIRECTION", False)
    monkeypatch.setattr(E, "SQUEEZE_REQUIRES_BULLISH", False)
    old = E.build_indicator_set(df, idx_close)
    new_adx = [k for k in new["criteria"] if k.startswith("ADX")][0]
    old_adx = [k for k in old["criteria"] if k.startswith("ADX")][0]
    # yeni kural eskisinin alt kümesidir (daha seçici)
    assert (new["criteria"][new_adx] <= old["criteria"][old_adx]).all()
    # değişmeyen kriterler birebir aynı
    for k in ("Trend: Fiyat > EMA20 > EMA50 (boğa dizilişi)", "MACD sinyal üzerinde, histogram güçleniyor"):
        assert (new["criteria"][k] == old["criteria"][k]).all()
