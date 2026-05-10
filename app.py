import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import math
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore") 

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="TradeEdge Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');
    html, body, [class*="css"] {
        font-family: 'DM Mono', monospace;
        background-color: #060810;
        color: #c8d0e0;
    }
    .block-container { padding: 1.2rem 2rem; }
    h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
    .main-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(135deg, #00f5a0, #00d9f5);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    .section-head {
        font-family: 'Syne', sans-serif;
        font-size: 1.05rem; font-weight: 700;
        color: #00f5a0; text-transform: uppercase;
        letter-spacing: 2px; border-bottom: 1px solid #1a2535;
        padding-bottom: 6px; margin: 22px 0 14px;
    }
    .signal-buy  { background: linear-gradient(135deg,#003d20,#005c2e); border:1px solid #00e676; border-radius:10px; padding:16px 20px; margin:10px 0; }
    .signal-sell { background: linear-gradient(135deg,#3d0011,#5c0019); border:1px solid #ff1744; border-radius:10px; padding:16px 20px; margin:10px 0; }
    .signal-neutral { background: linear-gradient(135deg,#1a1a2e,#16213e); border:1px solid #455a7a; border-radius:10px; padding:16px 20px; margin:10px 0; }
    .card { background:#0d1117; border:1px solid #1e2d42; border-radius:10px; padding:16px; margin:8px 0; }
    .profit-badge { display:inline-block; background:#003d20; color:#00e676; border-radius:6px; padding:4px 12px; font-weight:600; font-size:0.85rem; }
    .loss-badge   { display:inline-block; background:#3d0011; color:#ff5252; border-radius:6px; padding:4px 12px; font-weight:600; font-size:0.85rem; }
    div[data-testid="stMetricValue"] { font-family:'DM Mono',monospace!important; font-size:1.1rem!important; color:#e0eaff!important; }
    div[data-testid="stMetricLabel"] { font-family:'Syne',sans-serif!important; font-size:0.68rem!important; color:#6b8cba!important; text-transform:uppercase; letter-spacing:1px; }
    footer{display:none;} #MainMenu{display:none;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚡ TradeEdge Pro</div>', unsafe_allow_html=True)
st.markdown('<div style="color:#455a7a;font-size:0.8rem;margin-bottom:20px;font-family:DM Mono">9/15 EMA Signals · 200 DMA Delivery · Option Chain Greeks · Capital-Based P&L Advisor</div>', unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown('<div style="font-family:Syne;font-size:1rem;font-weight:700;color:#00f5a0;margin-bottom:14px">⚙ SETTINGS</div>', unsafe_allow_html=True)
    stock_input = st.text_input("Stock / Index Symbol", "NIFTY")
    trade_type  = st.selectbox("Mode", ["Intraday (5-min)", "Delivery (Daily)"])
    st.markdown("---")
    st.markdown('<div style="font-family:Syne;font-size:0.75rem;color:#6b8cba;text-transform:uppercase;letter-spacing:1px">💰 OPTION CAPITAL</div>', unsafe_allow_html=True)
    capital     = st.number_input("Available Capital (₹)", value=25000, step=1000, min_value=1000)
    risk_pct    = st.slider("Max Risk % per Trade", 5, 100, 30)
    option_type = st.selectbox("Option Preference", ["Best (Auto)", "CALL", "PUT"])
    expiry_days = st.selectbox("Expiry (days away)", [7, 15, 30, 45], index=1)
    st.markdown("---")
    uploaded_image = st.file_uploader("📸 Chart Screenshot (OCR)", type=["png","jpg","jpeg"])

# =====================================================
# SYMBOL RESOLVER
# =====================================================

def resolve_symbol(raw: str):
    raw = raw.strip().upper()
    index_map = {
        "NIFTY":     ("^NSEI",    "NIFTY 50",   50),
        "BANKNIFTY": ("^NSEBANK", "BANK NIFTY", 15),
        "SENSEX":    ("^BSESN",   "SENSEX",     10),
        "FINNIFTY":  ("^CNXFIN",  "FIN NIFTY",  40),
    }
    if raw in index_map:
        sym, name, lot = index_map[raw]
        return sym, name, lot, True
    if raw.endswith(".NS") or raw.endswith(".BO") or raw.startswith("^"):
        return raw, raw.replace(".NS","").replace(".BO",""), 1, False
    return raw + ".NS", raw, 1, False

ticker_sym, display_name, default_lot, is_index = resolve_symbol(stock_input)
lot_size = st.sidebar.number_input("Option Lot Size", value=default_lot, min_value=1)

# =====================================================
# DATA LOADERS
# =====================================================

@st.cache_data(ttl=180)
def load_intraday(sym):
    return yf.download(sym, period="5d", interval="5m", auto_adjust=True, progress=False)

@st.cache_data(ttl=3600)
def load_daily(sym):
    return yf.download(sym, period="1y", interval="1d", auto_adjust=True, progress=False)

def flatten(df, col):
    arr = np.array(df[col]).flatten()
    return pd.Series(arr, index=df.index, name=col)

# =====================================================
# BLACK-SCHOLES GREEKS
# =====================================================

def bs_greeks(S, K, T, r, sigma, opt_type="CALL"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return dict(price=0.01, delta=0.01, gamma=0, theta=0, vega=0, iv=sigma*100)
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    npd1 = norm.pdf(d1)
    if opt_type == "CALL":
        price = S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
    gamma = npd1 / (S*sigma*math.sqrt(T))
    theta = (-(S*npd1*sigma)/(2*math.sqrt(T)) -
             r*K*math.exp(-r*T)*(norm.cdf(d2) if opt_type=="CALL" else norm.cdf(-d2))) / 365
    vega  = S*npd1*math.sqrt(T)/100
    return dict(price=round(max(price,0.01),2), delta=round(delta,4),
                gamma=round(gamma,6), theta=round(theta,2),
                vega=round(vega,2), iv=round(sigma*100,1))

# =====================================================
# OPTION CHAIN BUILDER
# =====================================================

def build_option_chain(spot, expiry_days, hist_vol):
    T     = expiry_days / 365
    r     = 0.065
    sigma = hist_vol / 100
    step  = 100 if spot > 20000 else (50 if spot > 5000 else (20 if spot > 1000 else 5))
    atm   = round(spot / step) * step
    strikes = [atm + i*step for i in range(-10, 11)]
    rows = []
    np.random.seed(42)
    for K in strikes:
        moneyness = (spot - K) / spot
        iv_c = sigma * (1 + max(0,-moneyness)*0.5)
        iv_p = sigma * (1 + max(0, moneyness)*0.5 + 0.05)
        cg = bs_greeks(spot, K, T, r, iv_c, "CALL")
        pg = bs_greeks(spot, K, T, r, iv_p, "PUT")
        dist = abs(spot-K)/spot
        oi_base = max(500, int(50000*math.exp(-50*dist**2)))
        call_oi = max(100, int(oi_base*(1.2 if K>spot else 0.9)*(1+0.25*np.random.randn())))
        put_oi  = max(100, int(oi_base*(1.2 if K<spot else 0.9)*(1+0.25*np.random.randn())))
        tag = "ATM" if abs(K-spot)<step*0.6 else ("ITM" if K<spot else "OTM")
        rows.append({
            "Strike": K, "Tag": tag,
            "CALL_OI": call_oi, "CALL_Price": cg["price"],
            "CALL_Delta": cg["delta"], "CALL_Gamma": cg["gamma"],
            "CALL_Theta": cg["theta"], "CALL_Vega": cg["vega"], "CALL_IV": cg["iv"],
            "PUT_OI": put_oi, "PUT_Price": pg["price"],
            "PUT_Delta": pg["delta"], "PUT_Gamma": pg["gamma"],
            "PUT_Theta": pg["theta"], "PUT_Vega": pg["vega"], "PUT_IV": pg["iv"],
            "PCR": round(put_oi/call_oi, 2) if call_oi > 0 else 0,
        })
    return pd.DataFrame(rows)

# =====================================================
# OPTION RECOMMENDATION ENGINE
# =====================================================

def recommend_option(chain, spot, capital, risk_pct, lot_size, opt_type, expiry_days, signal, hist_vol):
    risk_amount = capital * risk_pct / 100
    if opt_type == "Best (Auto)":
        opt_type = "CALL" if signal == "BUY" else ("PUT" if signal == "SELL" else "CALL")
    col_p, col_d = f"{opt_type}_Price", f"{opt_type}_Delta"
    col_g, col_t = f"{opt_type}_Gamma", f"{opt_type}_Theta"
    col_v, col_oi, col_iv = f"{opt_type}_Vega", f"{opt_type}_OI", f"{opt_type}_IV"

    df = chain[chain["Tag"].isin(["ATM","OTM"])].copy()
    df["cost_lot"] = df[col_p] * lot_size
    df = df[df["cost_lot"] <= risk_amount]
    if df.empty:
        return None, opt_type

    df["sc_d"] = df[col_d].abs() / (df[col_d].abs().max() + 1e-9)
    df["sc_o"] = df[col_oi] / (df[col_oi].max() + 1e-9)
    df["sc_t"] = 1 - df[col_t].abs() / (df[col_t].abs().max() + 1e-9)
    df["score"]= 0.45*df["sc_d"] + 0.35*df["sc_o"] + 0.20*df["sc_t"]
    best = df.sort_values("score", ascending=False).iloc[0]

    premium   = float(best[col_p])
    strike    = float(best["Strike"])
    cost_lot  = premium * lot_size
    lots      = max(1, int(risk_amount // cost_lot))
    total_cost= lots * cost_lot

    T, r, sigma = expiry_days/365, 0.065, hist_vol/100
    expected_move = spot * sigma * math.sqrt(T)
    tgt_spot = spot + expected_move if opt_type=="CALL" else spot - expected_move
    sl_spot  = spot - expected_move*0.5 if opt_type=="CALL" else spot + expected_move*0.5
    target_T = max(T - 1/365, 0.001)

    tgt_g  = bs_greeks(tgt_spot, strike, target_T, r, sigma, opt_type)
    sl_g   = bs_greeks(sl_spot,  strike, target_T, r, sigma, opt_type)

    profit_pl = (tgt_g["price"] - premium) * lot_size
    loss_pl   = (sl_g["price"]  - premium) * lot_size
    rr = abs(profit_pl/loss_pl) if loss_pl != 0 else 0

    return {
        "opt_type": opt_type, "strike": strike, "tag": best["Tag"],
        "premium": round(premium,2), "lots": lots,
        "cost_lot": round(cost_lot,2), "total_cost": round(total_cost,2),
        "delta": round(float(best[col_d]),4), "gamma": round(float(best[col_g]),6),
        "theta": round(float(best[col_t]),2), "vega": round(float(best[col_v]),2),
        "iv": float(best[col_iv]), "oi": int(best[col_oi]),
        "target_opt": round(tgt_g["price"],2), "sl_opt": round(max(sl_g["price"],0),2),
        "total_profit": round(profit_pl*lots,0), "total_loss": round(loss_pl*lots,0),
        "profit_pct": round(profit_pl/cost_lot*100,1), "loss_pct": round(loss_pl/cost_lot*100,1),
        "rr": round(rr,2), "expected_move": round(expected_move,2),
        "target_spot": round(tgt_spot,2), "sl_spot": round(sl_spot,2),
    }, opt_type

# =====================================================
# MAIN
# =====================================================

try:
    is_intraday = trade_type.startswith("Intraday")

    with st.spinner(f"Loading {'5-min' if is_intraday else 'daily'} data for {display_name} …"):
        raw = load_intraday(ticker_sym) if is_intraday else load_daily(ticker_sym)

    if raw.empty:
        st.error(f"No data for **{ticker_sym}**.")
        st.stop()

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    dc  = "Datetime" if "Datetime" in raw.columns else "Date"
    raw.rename(columns={dc: "Date"}, inplace=True)
    raw.set_index("Date", inplace=True)
    raw = raw[~raw.index.duplicated()]
    df  = raw.copy()

    close_s = flatten(df, "Close")
    high_s  = flatten(df, "High")
    low_s   = flatten(df, "Low")
    open_s  = flatten(df, "Open")

    # ---- Indicators ----
    df["EMA9"]  = ta.trend.ema_indicator(close=close_s, window=9)
    df["EMA15"] = ta.trend.ema_indicator(close=close_s, window=15)
    df["EMA200"]= ta.trend.sma_indicator(close=close_s, window=min(200, len(df)-1))
    df["RSI"]   = ta.momentum.rsi(close=close_s, window=14)
    macd_obj    = ta.trend.MACD(close=close_s, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"]  = macd_obj.macd()
    df["MACD_S"]= macd_obj.macd_signal()
    df["MACD_H"]= macd_obj.macd_diff()
    bb          = ta.volatility.BollingerBands(close=close_s, window=20, window_dev=2)
    df["BB_U"]  = bb.bollinger_hband()
    df["BB_L"]  = bb.bollinger_lband()
    df.dropna(inplace=True)

    # Historical volatility
    log_ret  = np.log(flatten(df,"Close") / flatten(df,"Close").shift(1)).dropna()
    mult     = 252*78 if is_intraday else 252
    hist_vol = float(min(max(log_ret.std()*math.sqrt(mult)*100, 5.0), 120.0))

    close_s = flatten(df,"Close"); high_s = flatten(df,"High")
    low_s   = flatten(df,"Low");   open_s = flatten(df,"Open")

    lc   = float(close_s.iloc[-1])
    le9  = float(df["EMA9"].iloc[-1]);  le15 = float(df["EMA15"].iloc[-1])
    le200= float(df["EMA200"].iloc[-1]); lrsi = float(df["RSI"].iloc[-1])
    lmac = float(df["MACD"].iloc[-1]);  lmacs= float(df["MACD_S"].iloc[-1])
    lh   = float(high_s.iloc[-1]);      ll   = float(low_s.iloc[-1])

    win = min(20, len(df))
    support    = float(low_s.rolling(win).min().iloc[-1])
    resistance = float(high_s.rolling(win).max().iloc[-1])

    # ---- Signal logic ----
    if is_intraday:
        pe9 = float(df["EMA9"].iloc[-2]); pe15 = float(df["EMA15"].iloc[-2])
        cross_up = (pe9 < pe15) and (le9 > le15)
        cross_dn = (pe9 > pe15) and (le9 < le15)
        if cross_up or (lc > le9 and lc > le15):
            raw_signal = "BUY"
            signal_str = "📈 BUY SIGNAL — 9 EMA crossed above 15 EMA" if cross_up else "📈 BUY BIAS — Price above 9 & 15 EMA"
        elif cross_dn or (lc < le9 and lc < le15):
            raw_signal = "SELL"
            signal_str = "📉 SELL SIGNAL — 9 EMA crossed below 15 EMA" if cross_dn else "📉 SELL BIAS — Price below 9 & 15 EMA"
        else:
            raw_signal = "NEUTRAL"
            signal_str = "⏸ NEUTRAL — EMA convergence. Wait for clear cross."
    else:
        if lc > le200 and le9 > le15:
            raw_signal, signal_str = "BUY",  "📈 DELIVERY BUY — Price above 200 DMA + 9 EMA > 15 EMA"
        elif lc < le200 and le9 < le15:
            raw_signal, signal_str = "SELL", "📉 DELIVERY SELL — Price below 200 DMA + 9 EMA < 15 EMA"
        else:
            raw_signal, signal_str = "NEUTRAL", "⏸ MIXED SIGNALS — Wait for alignment"

    # ==================================================
    # OVERVIEW METRICS
    # ==================================================

    st.markdown(f'<div class="section-head">📊 {display_name} — Snapshot</div>', unsafe_allow_html=True)
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("Price",     f"₹{lc:,.2f}")
    m2.metric("9 EMA",     f"₹{le9:,.2f}")
    m3.metric("15 EMA",    f"₹{le15:,.2f}")
    m4.metric("200 DMA",   f"₹{le200:,.2f}")
    m5.metric("RSI (14)",  f"{lrsi:.1f}")
    m6.metric("Hist. Vol", f"{hist_vol:.1f}%")

    css = "signal-buy" if raw_signal=="BUY" else ("signal-sell" if raw_signal=="SELL" else "signal-neutral")
    st.markdown(f'<div class="{css}"><b style="font-size:1.05rem;font-family:Syne">{signal_str}</b></div>', unsafe_allow_html=True)

    # ==================================================
    # CANDLESTICK + INDICATORS CHART
    # ==================================================

    st.markdown(f'<div class="section-head">📉 {"5-Minute" if is_intraday else "Daily"} Chart — 9 EMA · 15 EMA · 200 DMA · MACD · RSI</div>', unsafe_allow_html=True)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.60, 0.20, 0.20], vertical_spacing=0.03,
        subplot_titles=("","RSI (14)","MACD"))

    fig.add_trace(go.Candlestick(x=df.index, open=open_s, high=high_s, low=low_s, close=close_s,
        name="Price", increasing_line_color="#00e676", decreasing_line_color="#ff1744",
        increasing_fillcolor="#00e676", decreasing_fillcolor="#ff1744"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["EMA9"],   mode="lines", name="EMA 9",
        line=dict(color="#ffeb3b", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA15"],  mode="lines", name="EMA 15",
        line=dict(color="#ff9800", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], mode="lines", name="200 DMA",
        line=dict(color="#40c4ff", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_U"], mode="lines", name="BB Upper",
        line=dict(color="#455a7a", width=1, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_L"], mode="lines", name="BB Lower",
        fill="tonexty", fillcolor="rgba(69,90,122,0.07)",
        line=dict(color="#455a7a", width=1, dash="dash")), row=1, col=1)

    fig.add_hline(y=support,    line_dash="dash", line_color="#69f0ae", line_width=1,
        annotation_text=f"S ₹{support:.0f}", annotation_font_color="#69f0ae", row=1, col=1)
    fig.add_hline(y=resistance, line_dash="dash", line_color="#ff5252", line_width=1,
        annotation_text=f"R ₹{resistance:.0f}", annotation_font_color="#ff5252", row=1, col=1)

    # EMA 9/15 cross signals
    e9s = df["EMA9"]; e15s = df["EMA15"]
    buy_x  = df.index[(e9s > e15s) & (e9s.shift(1) <= e15s.shift(1))]
    sell_x = df.index[(e9s < e15s) & (e9s.shift(1) >= e15s.shift(1))]
    if len(buy_x)>0:
        fig.add_trace(go.Scatter(x=buy_x, y=close_s[buy_x]*0.997, mode="markers+text",
            name="BUY Cross", text=["B"]*len(buy_x), textposition="bottom center",
            textfont=dict(color="#00e676",size=9),
            marker=dict(symbol="triangle-up",size=13,color="#00e676",
                        line=dict(color="white",width=1))), row=1, col=1)
    if len(sell_x)>0:
        fig.add_trace(go.Scatter(x=sell_x, y=close_s[sell_x]*1.003, mode="markers+text",
            name="SELL Cross", text=["S"]*len(sell_x), textposition="top center",
            textfont=dict(color="#ff1744",size=9),
            marker=dict(symbol="triangle-down",size=13,color="#ff1744",
                        line=dict(color="white",width=1))), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI",
        line=dict(color="#ce93d8",width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#ff5252", line_width=1, row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#69f0ae", line_width=1, row=2, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df["MACD_H"], name="MACD Hist",
        marker_color=["#00e676" if v>=0 else "#ff1744" for v in df["MACD_H"]], opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],   mode="lines", name="MACD",
        line=dict(color="#40c4ff",width=1.4)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_S"], mode="lines", name="Signal",
        line=dict(color="#ff9800",width=1.4)), row=3, col=1)

    fig.update_layout(template="plotly_dark", paper_bgcolor="#060810", plot_bgcolor="#0d1117",
        height=700, xaxis_rangeslider_visible=False,
        legend=dict(orientation="h",y=1.02,x=0,font=dict(size=10)),
        font=dict(family="DM Mono",color="#c8d0e0"),
        margin=dict(l=10,r=70,t=30,b=20))
    for i in range(1,4):
        fig.update_xaxes(showgrid=True, gridcolor="#0f1929", row=i, col=1,
            type="category" if is_intraday else "-", nticks=16)
        fig.update_yaxes(showgrid=True, gridcolor="#0f1929", side="right", row=i, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # ==================================================
    # LIVE TRADE SETUP
    # ==================================================

    st.markdown('<div class="section-head">🎯 Live Trade Setup — Underlying Entry</div>', unsafe_allow_html=True)
    atr = max(lh-ll, lc*0.005)
    be, bt, bs_ = lh+atr*0.1, lh+atr*1.5, ll-atr*0.2
    se, st_, ss  = ll-atr*0.1, ll-atr*1.5, lh+atr*0.2
    brr = round((bt-be)/(be-bs_),2) if (be-bs_)>0 else 0
    srr = round((se-st_)/(ss-se),2) if (ss-se)>0 else 0

    if raw_signal == "BUY":
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("🟢 BUY Entry",  f"₹{be:.2f}")
        c2.metric("🎯 Target",     f"₹{bt:.2f}", delta=f"+{((bt/be)-1)*100:.2f}%")
        c3.metric("🛑 Stop Loss",  f"₹{bs_:.2f}", delta=f"{((bs_/be)-1)*100:.2f}%")
        c4.metric("⚖️ R:R",         f"1 : {brr}")
    elif raw_signal == "SELL":
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("🔴 SELL Entry", f"₹{se:.2f}")
        c2.metric("🎯 Target",     f"₹{st_:.2f}", delta=f"{((st_/se)-1)*100:.2f}%")
        c3.metric("🛑 Stop Loss",  f"₹{ss:.2f}",  delta=f"+{((ss/se)-1)*100:.2f}%")
        c4.metric("⚖️ R:R",         f"1 : {srr}")
    else:
        st.info("⏸ Wait for a clear 9/15 EMA crossover signal before entering.")

    # ==================================================
    # OPTION CHAIN
    # ==================================================

    st.markdown('<div class="section-head">📋 Option Chain — OI · Delta · Gamma · Theta · Vega</div>', unsafe_allow_html=True)

    chain = build_option_chain(lc, expiry_days, hist_vol)
    tot_coi = chain["CALL_OI"].sum(); tot_poi = chain["PUT_OI"].sum()
    pcr_tot = round(tot_poi/tot_coi,2) if tot_coi else 0
    max_c_s = int(chain.loc[chain["CALL_OI"].idxmax(),"Strike"])
    max_p_s = int(chain.loc[chain["PUT_OI"].idxmax(), "Strike"])

    o1,o2,o3,o4 = st.columns(4)
    o1.metric("Total CALL OI", f"{tot_coi:,}")
    o2.metric("Total PUT OI",  f"{tot_poi:,}")
    o3.metric("PCR",           f"{pcr_tot}")
    o4.metric("Market Bias",   "Bullish 🐂" if pcr_tot>1 else "Bearish 🐻")

    cr, cs = st.columns(2)
    cr.info(f"🔴 Max CALL OI at ₹{max_c_s} → Key **Resistance**")
    cs.info(f"🟢 Max PUT OI at ₹{max_p_s} → Key **Support**")

    disp = chain[["Strike","Tag","CALL_OI","CALL_Price","CALL_Delta","CALL_Gamma",
                  "CALL_Theta","CALL_IV","PCR",
                  "PUT_OI","PUT_Price","PUT_Delta","PUT_Gamma","PUT_Theta","PUT_IV"]].copy()
    disp.columns = ["Strike","Type","C-OI","C-Price","C-Δ","C-Γ","C-Θ","C-IV%",
                    "PCR","P-OI","P-Price","P-Δ","P-Γ","P-Θ","P-IV%"]

    def hl(row):
        if row["Type"]=="ATM": return ['background-color:#0d2b3e;color:#40d9ff;font-weight:bold']*len(row)
        return ['']*len(row)

    st.dataframe(
        disp.style.apply(hl, axis=1).format({
            "Strike":"₹{:.0f}","C-OI":"{:,.0f}","C-Price":"₹{:.2f}",
            "C-Δ":"{:.3f}","C-Γ":"{:.5f}","C-Θ":"₹{:.2f}","C-IV%":"{:.1f}%",
            "PCR":"{:.2f}","P-OI":"{:,.0f}","P-Price":"₹{:.2f}",
            "P-Δ":"{:.3f}","P-Γ":"{:.5f}","P-Θ":"₹{:.2f}","P-IV%":"{:.1f}%",
        }),
        use_container_width=True, height=420
    )

    # OI Bar chart
    oi_fig = go.Figure()
    oi_fig.add_trace(go.Bar(x=chain["Strike"].astype(str), y=chain["CALL_OI"],
        name="CALL OI", marker_color="#ff5252", opacity=0.85))
    oi_fig.add_trace(go.Bar(x=chain["Strike"].astype(str), y=chain["PUT_OI"],
        name="PUT OI",  marker_color="#69f0ae", opacity=0.85))
    oi_fig.update_layout(template="plotly_dark", paper_bgcolor="#060810", plot_bgcolor="#0d1117",
        barmode="group", height=280, legend=dict(orientation="h",y=1.05),
        margin=dict(l=10,r=10,t=20,b=40),
        xaxis_title="Strike", yaxis_title="Open Interest",
        font=dict(family="DM Mono",size=11))
    st.plotly_chart(oi_fig, use_container_width=True)

    # ==================================================
    # OPTION RECOMMENDATION + P&L
    # ==================================================

    st.markdown('<div class="section-head">💡 Option Buying Advisor — Capital: ₹{:,}</div>'.format(capital), unsafe_allow_html=True)

    rec, chosen_type = recommend_option(
        chain, lc, capital, risk_pct, lot_size, option_type, expiry_days, raw_signal, hist_vol)

    if rec is None:
        st.error(f"❌ With ₹{capital:,} and {risk_pct}% risk limit (₹{int(capital*risk_pct/100):,}), "
                 f"no affordable strike found for 1 lot of {lot_size} units. "
                 f"Try increasing capital, raising risk %, or reducing lot size.")
    else:
        is_prof = rec["total_profit"] > 0 and rec["rr"] >= 1.2
        verdict  = "LIKELY PROFITABLE ✅" if is_prof else ("RISKY — LOW R:R ⚠️" if rec["rr"]>0 else "HIGH RISK ❌")
        bdge     = "profit-badge" if is_prof else "loss-badge"

        r1, r2 = st.columns([1.1, 1])

        with r1:
            st.markdown(f"""
<div class="card">
  <div style="font-family:Syne;font-size:1rem;font-weight:700;color:#00f5a0;margin-bottom:12px">
    🎯 {chosen_type} · Strike ₹{rec['strike']:.0f} ({rec['tag']}) · {expiry_days}d Expiry
  </div>
  <table style="width:100%;border-collapse:collapse;font-size:0.85rem">
    <tr><td style="color:#6b8cba;padding:5px 0">Option Premium</td>
        <td style="color:#e0eaff;text-align:right;font-weight:600">₹{rec['premium']}</td></tr>
    <tr><td style="color:#6b8cba;padding:5px 0">Cost per Lot ({lot_size} qty)</td>
        <td style="color:#e0eaff;text-align:right">₹{rec['cost_lot']:,.0f}</td></tr>
    <tr><td style="color:#6b8cba;padding:5px 0">Lots Recommended</td>
        <td style="color:#ffeb3b;text-align:right;font-weight:700;font-size:1rem">{rec['lots']}</td></tr>
    <tr><td style="color:#6b8cba;padding:5px 0">Total Investment</td>
        <td style="color:#40c4ff;text-align:right;font-weight:700">₹{rec['total_cost']:,.0f}</td></tr>
    <tr><td style="color:#6b8cba;padding:5px 0">Capital After Buy</td>
        <td style="color:#e0eaff;text-align:right">₹{capital - rec['total_cost']:,.0f}</td></tr>
  </table>
</div>""", unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
<div class="card">
  <div style="font-family:Syne;font-size:1rem;font-weight:700;color:#ffab40;margin-bottom:12px">
    📐 Option Greeks
  </div>
  <table style="width:100%;border-collapse:collapse;font-size:0.85rem">
    <tr><td style="color:#6b8cba;padding:5px 0">Delta (Δ)</td>
        <td style="color:#40c4ff;text-align:right;font-weight:600">{rec['delta']}</td></tr>
    <tr><td style="color:#6b8cba;padding:5px 0">Gamma (Γ)</td>
        <td style="color:#40c4ff;text-align:right">{rec['gamma']}</td></tr>
    <tr><td style="color:#6b8cba;padding:5px 0">Theta (Θ) / day</td>
        <td style="color:#ff5252;text-align:right;font-weight:600">₹{rec['theta']}</td></tr>
    <tr><td style="color:#6b8cba;padding:5px 0">Vega (V) per 1% IV</td>
        <td style="color:#e0eaff;text-align:right">₹{rec['vega']}</td></tr>
    <tr><td style="color:#6b8cba;padding:5px 0">Implied Volatility</td>
        <td style="color:#e0eaff;text-align:right">{rec['iv']:.1f}%</td></tr>
    <tr><td style="color:#6b8cba;padding:5px 0">Strike OI</td>
        <td style="color:#e0eaff;text-align:right">{rec['oi']:,}</td></tr>
  </table>
</div>""", unsafe_allow_html=True)

        # P&L table
        theta_drain = abs(rec['theta']) * lot_size * rec['lots']
        st.markdown(f"""
<div class="card" style="margin-top:12px">
  <div style="font-family:Syne;font-size:1rem;font-weight:700;color:#ce93d8;margin-bottom:12px">
    💰 P&L Projection — Based on 1σ Expected Move (±₹{rec['expected_move']:,.0f})
  </div>
  <div style="display:flex;gap:24px;flex-wrap:wrap;margin-bottom:14px;font-size:0.85rem">
    <div><div style="color:#6b8cba;font-size:0.7rem;text-transform:uppercase">Underlying Target</div>
         <div style="color:#00e676;font-weight:600">₹{rec['target_spot']:,.2f}</div></div>
    <div><div style="color:#6b8cba;font-size:0.7rem;text-transform:uppercase">Underlying SL</div>
         <div style="color:#ff5252;font-weight:600">₹{rec['sl_spot']:,.2f}</div></div>
    <div><div style="color:#6b8cba;font-size:0.7rem;text-transform:uppercase">Option at Target</div>
         <div style="color:#00e676;font-weight:600">₹{rec['target_opt']}</div></div>
    <div><div style="color:#6b8cba;font-size:0.7rem;text-transform:uppercase">Option at SL</div>
         <div style="color:#ff5252;font-weight:600">₹{rec['sl_opt']}</div></div>
    <div><div style="color:#6b8cba;font-size:0.7rem;text-transform:uppercase">Theta Drain / Day</div>
         <div style="color:#ff9800;font-weight:600">₹{theta_drain:,.0f}</div></div>
  </div>
  <table style="width:100%;border-collapse:collapse;font-size:0.9rem">
    <tr style="border-bottom:1px solid #1e2d42">
      <td style="padding:8px 4px;color:#6b8cba">Scenario</td>
      <td style="padding:8px 4px;color:#6b8cba;text-align:right">Per Lot</td>
      <td style="padding:8px 4px;color:#6b8cba;text-align:right">{rec['lots']} Lot(s) Total</td>
      <td style="padding:8px 4px;color:#6b8cba;text-align:right">Return %</td>
    </tr>
    <tr>
      <td style="padding:9px 4px;color:#00e676">🎯 TARGET HIT</td>
      <td style="padding:9px 4px;color:#00e676;text-align:right">₹{rec['total_profit']/rec['lots']:,.0f}</td>
      <td style="padding:9px 4px;color:#00e676;text-align:right;font-weight:700;font-size:1rem">₹{rec['total_profit']:,.0f}</td>
      <td style="padding:9px 4px;color:#00e676;text-align:right">{rec['profit_pct']:.1f}%</td>
    </tr>
    <tr>
      <td style="padding:9px 4px;color:#ff5252">🛑 STOP LOSS HIT</td>
      <td style="padding:9px 4px;color:#ff5252;text-align:right">₹{rec['total_loss']/rec['lots']:,.0f}</td>
      <td style="padding:9px 4px;color:#ff5252;text-align:right;font-weight:700;font-size:1rem">₹{rec['total_loss']:,.0f}</td>
      <td style="padding:9px 4px;color:#ff5252;text-align:right">{rec['loss_pct']:.1f}%</td>
    </tr>
    <tr style="border-top:1px solid #1e2d42">
      <td colspan="2" style="padding:9px 4px;color:#6b8cba">Risk : Reward Ratio</td>
      <td colspan="2" style="padding:9px 4px;color:#e0eaff;text-align:right;font-weight:700;font-size:1.05rem">1 : {rec['rr']}</td>
    </tr>
  </table>
  <div style="margin-top:14px">
    <span class="{bdge}">{verdict}</span>
    <span style="color:#6b8cba;font-size:0.75rem;margin-left:14px">
      Signal: {raw_signal} · IV: {rec['iv']:.1f}% · Hist.Vol: {hist_vol:.1f}% ·
      Time decay: ₹{theta_drain:,.0f}/day
    </span>
  </div>
</div>""", unsafe_allow_html=True)

        # Greeks explainer
        with st.expander("📖 Greeks & OI Explained — What This Means for Your Trade"):
            st.markdown(f"""
### 📐 Delta ({rec['delta']})
For every **₹1 move** in {display_name}, your option moves ≈ **₹{abs(rec['delta']):.2f}**.
With {rec['lots']} lot(s) × {lot_size} qty → a **₹100 move** in the stock = **₹{abs(rec['delta'])*100*lot_size*rec['lots']:,.0f}** P&L.
- Delta near 0.5 = ATM option (most sensitive, balanced risk)
- Delta near 0.2-0.3 = OTM (cheaper, needs bigger move)
- Delta near 0.8+ = ITM (expensive but moves almost like stock)

### 📐 Gamma ({rec['gamma']})
Delta accelerates by {rec['gamma']} for each ₹1 spot move. High Gamma near ATM means your option gains speed as it moves ITM. Great for intraday!

### ⏱ Theta (₹{rec['theta']}/day per unit)
**You LOSE ₹{theta_drain:,.0f} per day** just sitting still (across all lots).
{"⚠️ **URGENT**: {expiry_days} days to expiry = very fast theta decay. Exit quickly if the move doesn't happen within 1-2 sessions!" if expiry_days <= 7 else f"You have {expiry_days} days — theta is manageable but still exits if stuck."}

### 📈 Vega (₹{rec['vega']}/unit per 1% IV)
If implied volatility rises 1%, your option gains ₹{rec['vega']*lot_size*rec['lots']:,.0f} total (across all lots).
- Buy before expected volatility events (results, budget, RBI policy)
- Avoid buying when IV is already very high (expensive options)

### 📊 Open Interest & PCR
- **Max Call OI at ₹{max_c_s}** = Strong resistance. Market makers have sold calls here → price may struggle to break above.
- **Max Put OI at ₹{max_p_s}** = Strong support. Market makers have sold puts here → price may bounce.
- **PCR = {pcr_tot}** → {"PCR > 1 = More puts written = Bullish bias (market expected to stay above support)" if pcr_tot > 1 else "PCR < 1 = More calls written = Bearish bias (market expected to stay below resistance)"}

### ✅ Trade Checklist
- [ ] Signal is {raw_signal} on 9/15 EMA cross
- [ ] Buying {chosen_type} because {"price moving UP" if chosen_type=="CALL" else "price moving DOWN"}
- [ ] Strike ₹{rec['strike']:.0f} ({rec['tag']}) with good OI ({rec['oi']:,}) = liquid
- [ ] Risk ₹{rec['total_cost']:,.0f} ({risk_pct}% of capital)
- [ ] Exit at ₹{rec['target_opt']} (profit) or ₹{rec['sl_opt']} (stop loss)
- [ ] Don't hold beyond {max(1, expiry_days-2)} days from now
""")

    # ==================================================
    # DELIVERY SCORECARD
    # ==================================================

    if not is_intraday:
        st.markdown('<div class="section-head">📦 Delivery Scorecard — 200 DMA Strategy</div>', unsafe_allow_html=True)
        checks = [
            (lc > le200,        "Price above 200 DMA (primary trend is UP)"),
            (le9 > le15,        "9 EMA above 15 EMA (short-term momentum bullish)"),
            (40 < lrsi < 70,    "RSI in healthy zone 40–70 (not overbought/oversold)"),
            (lmac > lmacs,      "MACD above Signal line (momentum positive)"),
            (lc > resistance*0.98, "Price near/above resistance (breakout zone)"),
        ]
        score = sum(1 for c, _ in checks if c)
        for chk, label in checks:
            st.write(("✅ " if chk else "❌ ") + label)
        if score >= 4:
            st.success(f"✅ Strong Delivery Buy (Score {score}/5) — Accumulate near 9 or 15 EMA on dips")
        elif score >= 2:
            st.warning(f"⚠️ Moderate Setup (Score {score}/5) — Wait for 200 DMA + EMA alignment")
        else:
            st.error(f"❌ Weak Setup (Score {score}/5) — Avoid. Price below 200 DMA = downtrend")

    # ==================================================
    # SCREENSHOT OCR
    # ==================================================

    if uploaded_image is not None:
        st.markdown('<div class="section-head">🤖 Screenshot OCR Analysis</div>', unsafe_allow_html=True)
        from PIL import Image
        import cv2, easyocr
        image = Image.open(uploaded_image)
        st.image(image, use_container_width=True)
        uploaded_image.seek(0)
        fb = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        ocv = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        with st.spinner("Running OCR …"):
            reader = easyocr.Reader(["en"], gpu=False)
            res = reader.readtext(ocv)
        extracted = " ".join([t[1] for t in res])
        st.write(extracted if extracted.strip() else "No text detected in image.")

    # ==================================================
    # RAW DATA
    # ==================================================

    with st.expander("📋 Raw OHLCV + Indicators (last 40 rows)"):
        cols = ["Open","High","Low","Close","Volume","EMA9","EMA15","EMA200","RSI","MACD","BB_U","BB_L"]
        sc   = [c for c in cols if c in df.columns]
        st.dataframe(df[sc].tail(40), use_container_width=True)

except Exception as e:
    st.error(f"⚠️ Error: {e}")
    import traceback
    st.code(traceback.format_exc())
