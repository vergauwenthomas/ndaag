#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NDAAG — Technical Analysis Dashboard
=====================================

A lightweight, interactive dashboard for stock technical analysis using the
ndaag package.  Built with Streamlit and Plotly.

Run locally:
    streamlit run dashboard/app.py

Host for free on Streamlit Community Cloud — see dashboard/README.md.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import ndaag

# ---------------------------------------------------------------------------
#  Google Sheet — user profiles
# ---------------------------------------------------------------------------
SHEET_ID = "1Ih-pPV-V4jtDRecsoEe5_tveb0LVNzxAVb1Gk4nfe0I"
SHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"


@st.cache_data(ttl=300, show_spinner="Loading user profiles …")
def load_user_profiles() -> dict[str, list[str]]:
    """Fetch the public Google Sheet and return {username: [tickers …]}."""
    df = pd.read_csv(SHEET_CSV_URL, header=None)
    profiles: dict[str, list[str]] = {}
    for col in df.columns:
        name = str(df.iloc[0, col]).strip()
        tickers = (
            df.iloc[1:, col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
            .tolist()
        )
        if name:
            profiles[name] = tickers
    return profiles


# ---------------------------------------------------------------------------
#  Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NDAAG — Technical Analysis",
    page_icon="📈",
    layout="wide",
)

st.title("📈 NDAAG — Ons eigen dashboard")
st.markdown(
    "Welkom bij het NDAAG technische analyse dashboard! Dit dashboard kan inzicht geven in koop/verkoop signalen."
)

# ---------------------------------------------------------------------------
#  Sidebar — user inputs
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")

# --- User profile ---
st.sidebar.subheader("👤 User Profile")
try:
    profiles = load_user_profiles()
    user_names = list(profiles.keys())
except Exception:
    profiles = {}
    user_names = []
    st.sidebar.warning("Could not load user profiles from Google Sheet.")

selected_user = st.sidebar.selectbox(
    "Selecteer je naam",
    options=["— geen —"] + user_names,
    index=0,
    help="Kies je naam om je persoonlijke watchlist te laden.",
)

st.sidebar.divider()

# --- Stock selection ---
st.sidebar.subheader("📊 Stock")

if selected_user != "— geen —" and selected_user in profiles:
    user_stocks = profiles[selected_user]
    if user_stocks:
        ticker = st.sidebar.selectbox(
            "Kies een aandeel uit je watchlist",
            options=user_stocks,
            index=0,
        )
    else:
        st.sidebar.info("Je watchlist is leeg. Voer handmatig een ticker in.")
        ticker = st.sidebar.text_input("Kies een aandeel", value="AAPL")
else:
    ticker = st.sidebar.text_input("Kies een aandeel", value="AAPL")

period = st.sidebar.selectbox(
    "History period",
    options=["3mo", "6mo", "9mo", "1y", "2y", "5y"],
    index=0,
    help="Amount of historical data to download.",
)

st.sidebar.divider()

# --- Display options ---
st.sidebar.subheader("🎨 Display")
show_buy = st.sidebar.checkbox("Show buy signals", value=True)
show_sell = st.sidebar.checkbox("Show sell signals", value=True)

st.sidebar.divider()

# --- Indicator settings (collapsible) ---
with st.sidebar.expander("⚙️ Momentum Settings", expanded=False):
    mom_window = st.number_input(
        "Window (days)",
        min_value=1,
        max_value=60,
        value=10,
        step=1,
        help="Look-back window in days for the momentum calculation.",
        key="mom_window"
    )
    mom_threshold = st.number_input(
        "Signal threshold",
        min_value=-50.0,
        max_value=50.0,
        value=0.0,
        step=0.5,
        help="Momentum must cross this threshold to trigger a signal.",
        key="mom_threshold"
    )

with st.sidebar.expander("⚙️ MACD Settings", expanded=False):
    col_macd1, col_macd2 = st.columns(2)
    macd_fast = col_macd1.number_input("Fast EMA", min_value=2, max_value=50, value=12, step=1, key="macd_fast")
    macd_slow = col_macd2.number_input("Slow EMA", min_value=2, max_value=100, value=26, step=1, key="macd_slow")
    macd_signal = st.number_input(
        "Signal EMA", min_value=2, max_value=50, value=9, step=1, key="macd_signal"
    )
    macd_early = st.number_input(
        "Early-warning threshold",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.1,
        help="Set > 0 to receive buy/sell warnings *before* the actual crossover.",
        key="macd_early"
    )

with st.sidebar.expander("⚙️ CCI Settings", expanded=False):
    cci_period = st.number_input(
        "Period", min_value=5, max_value=100, value=20, step=1, key="cci_period"
    )
    col_cci1, col_cci2 = st.columns(2)
    cci_buy_thresh = col_cci1.number_input(
        "Buy threshold", min_value=-300, max_value=0, value=-100, step=10, key="cci_buy"
    )
    cci_sell_thresh = col_cci2.number_input(
        "Sell threshold", min_value=0, max_value=300, value=100, step=10, key="cci_sell"
    )

st.sidebar.divider()

# --- Composite score settings ---
with st.sidebar.expander("📊 Composite Score Settings", expanded=False):
    show_score = st.checkbox(
        "Show composite buy/sell score",
        value=False,
        help="Overlay a weighted buy/sell score (0–100 %) on the price chart.",
        key="show_score",
    )
    score_window = st.number_input(
        "Rolling window (days)",
        min_value=1,
        max_value=7,
        value=3,
        step=1,
        help="Backward-looking window over which signals are aggregated.",
        key="score_window",
    )
    st.markdown("**Indicator weights** (will be normalised to 1)")
    w_mom = st.slider("Momentum", 0.0, 1.0, 0.2, 0.01, key="w_mom")
    w_macd = st.slider("MACD", 0.0, 1.0, 0.6, 0.01, key="w_macd")
    w_cci = st.slider("CCI", 0.0, 1.0, 0.2, 0.01, key="w_cci")

st.sidebar.divider()

# --- Go button ---
run_analysis = st.sidebar.button("🔄 Calculate", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
#  Core analysis (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Fetching stock data …")
def fetch_stock(ticker: str, period: str) -> pd.DataFrame:
    """Download stock data via yfinance and return the DataFrame."""
    stock = ndaag.Stock(ticker)
    stock.fetch_data(period=period)
    return stock.data


def run_indicators(
    data: pd.DataFrame,
    # Momentum params
    mom_window: int,
    mom_threshold: float,
    # MACD params
    macd_fast: int,
    macd_slow: int,
    macd_signal_period: int,
    macd_early: float,
    # CCI params
    cci_period: int,
    cci_buy_thresh: float,
    cci_sell_thresh: float,
) -> dict:
    """
    Run all three technical indicators on *data* and return a dict with 
    computed series and signal timestamps.
    """
    # Build a lightweight Stock-like object so we can feed it to the indices
    # (they only read `stock.data`).
    class _StockProxy:
        def __init__(self, df):
            self.data = df

    proxy = _StockProxy(data)

    # --- Momentum ---
    mom = ndaag.MomentumIndex(
        stock=proxy,
        window=f"{mom_window}d",
        interpolate_weekend=True,
    )
    mom.calculate()
    mom_buy = mom.find_buy_signal(threshold=mom_threshold, require_rising=True)
    mom_sell = mom.find_sell_signal(threshold=-abs(mom_threshold), require_falling=True)

    # --- MACD ---
    macd = ndaag.MACDIndex(
        stock=proxy,
        ema_fast=macd_fast,
        ema_slow=macd_slow,
        ema_signal=macd_signal_period,
    )
    macd.calculate()
    macd_buy = macd.find_buy_signal(early_warning_threshold=macd_early)
    macd_sell = macd.find_sell_signal(early_warning_threshold=macd_early)

    # --- CCI ---
    cci = ndaag.CCIIndex(stock=proxy, period=cci_period)
    cci.calculate()
    cci_buy = cci.find_buy_signal(threshold=cci_buy_thresh)
    cci_sell = cci.find_sell_signal(threshold=cci_sell_thresh)

    return dict(
        # Momentum
        mom_values=mom.momentum,
        mom_buy=mom_buy,
        mom_sell=mom_sell,
        # MACD
        macd_line=macd.macd_line,
        signal_line=macd.signal_line,
        histogram=macd.histogram,
        macd_buy=macd_buy,
        macd_sell=macd_sell,
        # CCI
        cci_values=cci.cci,
        cci_buy=cci_buy,
        cci_sell=cci_sell,
    )


# ---------------------------------------------------------------------------
#  Composite buy / sell score
# ---------------------------------------------------------------------------
def compute_composite_scores(
    data: pd.DataFrame,
    ind: dict,
    window: int,
    w_mom: float,
    w_macd: float,
    w_cci: float,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute a composite buy and sell score (0–100 %) over time.

    For each indicator a binary series is created (1 on signal dates, 0
    elsewhere).  A backward-looking rolling window counts how many signals
    fell inside the window, then divides by the window size to obtain a
    density (0–1).  The three densities are combined with user-supplied
    weights that are normalised to sum to 1.

    Returns
    -------
    buy_score, sell_score : pd.Series
        Values in [0, 100].
    """
    idx = data.index

    def _binary(signal_idx: pd.DatetimeIndex) -> pd.Series:
        s = pd.Series(0.0, index=idx)
        overlap = signal_idx.intersection(idx)
        if len(overlap):
            s.loc[overlap] = 1.0
        return s

    # Binary signal series
    buy_series = {
        "mom": _binary(ind["mom_buy"]),
        "macd": _binary(ind["macd_buy"]),
        "cci": _binary(ind["cci_buy"]),
    }
    sell_series = {
        "mom": _binary(ind["mom_sell"]),
        "macd": _binary(ind["macd_sell"]),
        "cci": _binary(ind["cci_sell"]),
    }

    # Normalise weights
    total_w = w_mom + w_macd + w_cci
    if total_w == 0:
        total_w = 1.0  # avoid division by zero
    weights = {
        "mom": w_mom / total_w,
        "macd": w_macd / total_w,
        "cci": w_cci / total_w,
    }

    def _score(series_dict: dict[str, pd.Series]) -> pd.Series:
        score = pd.Series(0.0, index=idx)
        for key, s in series_dict.items():
            density = s.rolling(window, min_periods=1).mean()
            score += weights[key] * density
        return score * 100.0  # convert to %

    return _score(buy_series), _score(sell_series)


# ---------------------------------------------------------------------------
#  Build Plotly figure
# ---------------------------------------------------------------------------
def build_figure(data: pd.DataFrame, ind: dict, ticker: str,
                 show_buy: bool, show_sell: bool,
                 cci_buy_thresh: float, cci_sell_thresh: float,
                 buy_score: pd.Series | None = None,
                 sell_score: pd.Series | None = None) -> go.Figure:
    """
    Create a multi-panel Plotly figure replicating the developscript dashboard.

    Panels (top → bottom):
        1. Stock closing price  (with buy markers)
        2. Volume histogram
        3. MACD  (line, signal, histogram)
        4. CCI (Commodity Channel Index)
        5. Momentum

    """
    # When scores are shown, use a secondary y-axis on the price panel
    has_scores = buy_score is not None and sell_score is not None
    specs = [[{"secondary_y": True}]] + [[{"secondary_y": False}]] * 4

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.30, 0.08, 0.30, 0.16, 0.16],
        subplot_titles=[
            f"{ticker} — Close Price",
            "Volume",
            "MACD",
            "CCI (Commodity Channel Index)",
            "Momentum",
        ],
        specs=specs,
    )

    # ---- 1. Price --------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=data.index, y=data["Close"],
            mode="lines", name="Close",
            line=dict(color="black", width=2),
        ),
        row=1, col=1,
    )

    # Buy/sell markers on the price chart
    if show_buy:
        for label, idx, color, symbol in [
            ("Mom Buy", ind["mom_buy"], "green", "triangle-up"),
            ("MACD Buy", ind["macd_buy"], "blue", "triangle-up"),
            ("CCI Buy", ind["cci_buy"], "orange", "triangle-up"),
        ]:
            if len(idx) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=idx,
                        y=data.loc[idx, "Close"],
                        mode="markers",
                        name=label,
                        marker=dict(symbol=symbol, size=10, color=color, 
                                    line=dict(width=1, color="white")),
                    ),
                    row=1, col=1,
                )

    if show_sell:
        for label, idx, color, symbol in [
            ("Mom Sell", ind["mom_sell"], "red", "triangle-down"),
            ("MACD Sell", ind["macd_sell"], "purple", "triangle-down"),
            ("CCI Sell", ind["cci_sell"], "crimson", "triangle-down"),
        ]:
            if len(idx) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=idx,
                        y=data.loc[idx, "Close"],
                        mode="markers",
                        name=label,
                        marker=dict(symbol=symbol, size=10, color=color,
                                    line=dict(width=1, color="white")),
                    ),
                    row=1, col=1,
                )

    # ---- Score overlay on price panel ------------------------------------
    if has_scores:
        fig.add_trace(
            go.Scatter(
                x=buy_score.index, y=buy_score.values,
                mode="lines", name="Buy Score (%)",
                line=dict(color="green", width=2, dash="dot"),
                opacity=0.8,
            ),
            row=1, col=1, secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=sell_score.index, y=sell_score.values,
                mode="lines", name="Sell Score (%)",
                line=dict(color="red", width=2, dash="dot"),
                opacity=0.8,
            ),
            row=1, col=1, secondary_y=True,
        )
        fig.update_yaxes(
            title_text="Score (%)", secondary_y=True, row=1, col=1,
            range=[0, 100], showgrid=False,
        )

    # ---- 2. Volume -------------------------------------------------------
    fig.add_trace(
        go.Bar(
            x=data.index, y=data["Volume"],
            name="Volume",
            marker_color="grey", opacity=0.5,
            showlegend=False,
        ),
        row=2, col=1,
    )
    
     # ---- 3. MACD ---------------------------------------------------------
    macd_line = ind["macd_line"].dropna()
    signal_line = ind["signal_line"].dropna()
    histogram = ind["histogram"].dropna()

    fig.add_trace(
        go.Scatter(
            x=macd_line.index, y=macd_line.values,
            mode="lines", name="MACD Line",
            line=dict(color="blue", width=1.5),
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=signal_line.index, y=signal_line.values,
            mode="lines", name="Signal Line",
            line=dict(color="red", width=1.5),
        ),
        row=3, col=1,
    )
    # Histogram bars coloured by sign
    hist_colors = ["green" if v >= 0 else "red" for v in histogram.values]
    fig.add_trace(
        go.Bar(
            x=histogram.index, y=histogram.values,
            name="Histogram",
            marker_color=hist_colors, opacity=0.35,
        ),
        row=3, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5,
                  row=3, col=1)

    _add_signal_vlines(fig, ind["macd_buy"], ind["macd_sell"],
                       show_buy, show_sell, row=3)

    # ---- 4. CCI ----------------------------------------------------------
    cci = ind["cci_values"].dropna()
    fig.add_trace(
        go.Scatter(
            x=cci.index, y=cci.values,
            mode="lines", name="CCI",
            line=dict(color="royalblue", width=1.5),
        ),
        row=4, col=1,
    )
    # Overbought / oversold reference lines
    fig.add_hline(y=cci_sell_thresh, line_dash="dash", line_color="red",
                  annotation_text=f"+{cci_sell_thresh}", row=4, col=1)
    fig.add_hline(y=cci_buy_thresh, line_dash="dash", line_color="green",
                  annotation_text=str(cci_buy_thresh), row=4, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3,
                  row=4, col=1)

    # Buy/sell vertical lines on indicator panels
    _add_signal_vlines(fig, ind["cci_buy"], ind["cci_sell"],
                       show_buy, show_sell, row=4)

    # ---- 5. Momentum -----------------------------------------------------
    mom = ind["mom_values"].dropna()
    fig.add_trace(
        go.Scatter(
            x=mom.index, y=mom.values,
            mode="lines", name="Momentum",
            line=dict(color="purple", width=1.5),
        ),
        row=5, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5,
                  row=5, col=1)

    _add_signal_vlines(fig, ind["mom_buy"], ind["mom_sell"],
                       show_buy, show_sell, row=5)

   

    # ---- Layout tweaks ---------------------------------------------------
    fig.update_layout(
        height=1100,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(size=10),
        ),
        margin=dict(l=50, r=20, t=80, b=30),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="CCI", row=4, col=1)
    fig.update_yaxes(title_text="Momentum", row=5, col=1)
    fig.update_xaxes(title_text="Date", row=5, col=1)

    return fig


def _add_signal_vlines(
    fig: go.Figure,
    buy_idx: pd.DatetimeIndex,
    sell_idx: pd.DatetimeIndex,
    show_buy: bool,
    show_sell: bool,
    row: int,
) -> None:
    """Add semi-transparent vertical lines for buy/sell signals to a subplot."""
    if show_buy:
        for ts in buy_idx:
            fig.add_vline(
                x=ts, line_width=1.5, line_dash="solid",
                line_color="green", opacity=0.45, row=row, col=1,
            )
    if show_sell:
        for ts in sell_idx:
            fig.add_vline(
                x=ts, line_width=1.5, line_dash="solid",
                line_color="red", opacity=0.45, row=row, col=1,
            )


# ---------------------------------------------------------------------------
#  Main logic — run on button press or first load
# ---------------------------------------------------------------------------
# Use session_state to persist results across reruns
if "data" not in st.session_state:
    st.session_state["data"] = None
    st.session_state["indicators"] = None
    st.session_state["ticker_used"] = None

if run_analysis or st.session_state["data"] is None:
    try:
        with st.spinner(f"Fetching **{ticker.upper()}** data …"):
            stock_data = fetch_stock(ticker.upper(), period)

        if stock_data.empty:
            st.error("No data returned.  Check the ticker symbol and try again.")
            st.stop()

        with st.spinner("Calculating indicators …"):
            indicators = run_indicators(
                data=stock_data,
                mom_window=mom_window,
                mom_threshold=mom_threshold,
                macd_fast=macd_fast,
                macd_slow=macd_slow,
                macd_signal_period=macd_signal,
                macd_early=macd_early,
                cci_period=cci_period,
                cci_buy_thresh=cci_buy_thresh,
                cci_sell_thresh=cci_sell_thresh,
            )

        st.session_state["data"] = stock_data
        st.session_state["indicators"] = indicators
        st.session_state["ticker_used"] = ticker.upper()

    except Exception as exc:
        st.error(f"Error: {exc}")
        st.stop()

# If we have data, render the dashboard
if st.session_state["data"] is not None:
    data = st.session_state["data"]
    ind = st.session_state["indicators"]
    used_ticker = st.session_state["ticker_used"]

    # ---- Summary metrics row ----
    col1, col2, col3, col4 = st.columns(4)
    latest_close = data["Close"].iloc[-1]
    price_change = data["Close"].iloc[-1] - data["Close"].iloc[-2]
    pct_change = price_change / data["Close"].iloc[-2] * 100

    col1.metric(
        f"{used_ticker} Close",
        f"${latest_close:,.2f}",
        f"{price_change:+,.2f} ({pct_change:+.2f}%)",
    )
    col2.metric("Momentum Buy / Sell", f"{len(ind['mom_buy'])} / {len(ind['mom_sell'])}")
    col3.metric("MACD Buy / Sell", f"{len(ind['macd_buy'])} / {len(ind['macd_sell'])}")
    col4.metric("CCI Buy / Sell", f"{len(ind['cci_buy'])} / {len(ind['cci_sell'])}")

    # ---- Composite scores ----
    buy_score, sell_score = None, None
    if show_score:
        buy_score, sell_score = compute_composite_scores(
            data, ind,
            window=score_window,
            w_mom=w_mom,
            w_macd=w_macd,
            w_cci=w_cci,
        )

    # ---- Chart ----
    fig = build_figure(
        data, ind, used_ticker,
        show_buy=show_buy, show_sell=show_sell,
        cci_buy_thresh=cci_buy_thresh,
        cci_sell_thresh=cci_sell_thresh,
        buy_score=buy_score,
        sell_score=sell_score,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Signal tables (collapsible) ----
    with st.expander("📋 Signal details"):
        tab_mom, tab_macd, tab_cci = st.tabs(["Momentum", "MACD", "CCI"])

        def _signals_df(buy_idx, sell_idx):
            """Create a tidy DataFrame listing all signals."""
            rows = []
            for ts in buy_idx:
                rows.append({"Date": ts, "Signal": "BUY", "Price": data.loc[ts, "Close"]})
            for ts in sell_idx:
                rows.append({"Date": ts, "Signal": "SELL", "Price": data.loc[ts, "Close"]})
            if not rows:
                return pd.DataFrame(columns=["Date", "Signal", "Price"])
            df = pd.DataFrame(rows).sort_values("Date", ascending=False).reset_index(drop=True)
            df["Price"] = df["Price"].map("${:,.2f}".format)
            return df

        with tab_mom:
            st.dataframe(_signals_df(ind["mom_buy"], ind["mom_sell"]),
                         use_container_width=True, hide_index=True)
        with tab_macd:
            st.dataframe(_signals_df(ind["macd_buy"], ind["macd_sell"]),
                         use_container_width=True, hide_index=True)
        with tab_cci:
            st.dataframe(_signals_df(ind["cci_buy"], ind["cci_sell"]),
                         use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
#  Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Built with [Streamlit](https://streamlit.io) · "
    "Powered by [ndaag](https://github.com/vergauwenthomas/ndaag)"
)
