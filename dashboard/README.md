# NDAAG — Technical Analysis Dashboard

A lightweight, interactive stock-analysis dashboard built with
**Streamlit** + **Plotly**, powered by the `ndaag` package.

![screenshot](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)

---

## Features

| Control | What it does |
|---|---|
| **Ticker symbol** | Analyse any stock listed on Yahoo Finance |
| **History period** | Choose how far back to look (3 months → 5 years) |
| **Momentum window** | Days used for momentum calculation |
| **MACD parameters** | Fast EMA, Slow EMA, Signal EMA, Early-warning threshold |
| **CCI period & thresholds** | Overbought / oversold levels |
| **Show buy / sell signals** | Toggle signal markers on the chart |
| **Calculate button** | Re-run all indicators with current settings |

---

## Run Locally

```bash
# 1. Install dependencies (from the repository root)
pip install -e .                       # install ndaag in editable mode
pip install -r dashboard/requirements.txt

# 2. Launch the dashboard
streamlit run dashboard/app.py
```

The app opens at **http://localhost:8501**.

---

## Host on Streamlit Community Cloud (free)

Streamlit Community Cloud is the easiest way to host this dashboard online,
directly from the GitHub repository — no server setup required.

### Step-by-step

1. **Push this repository** (with the `dashboard/` folder) to GitHub.

2. **Go to** [share.streamlit.io](https://share.streamlit.io) and sign in with
   your GitHub account.

3. Click **"New app"** and fill in:

   | Field | Value |
   |---|---|
   | **Repository** | `vergauwenthomas/ndaag` |
   | **Branch** | `main` |
   | **Main file path** | `dashboard/app.py` |

4. Under **Advanced settings → Python version**, select `3.11` (or later).

5. Under **Advanced settings → Custom packages / Secrets**, add any API keys
   if needed (not required for the basic dashboard).

6. Click **Deploy!**

Streamlit Cloud will:
- Install the packages listed in `dashboard/requirements.txt`
- Install the `ndaag` package from the repo root (it auto-detects `pyproject.toml`)
- Start the app and give you a public URL, e.g.  
  `https://vergauwenthomas-ndaag-dashboardapp-xxxxx.streamlit.app`

### How Streamlit Cloud finds the dependencies

Streamlit Cloud looks for a `requirements.txt` **in the same folder** as the
main file (`dashboard/app.py`).  It also auto-installs the repo itself when it
finds a `pyproject.toml` or `setup.py` at the repo root.  No extra
configuration is needed.

### Keeping it up to date

Every time you push to the `main` branch, Streamlit Cloud will **automatically
redeploy** the latest version.

---

## Alternative: GitHub Pages

GitHub Pages only serves **static** files (HTML / JS / CSS) and cannot run
Python.  If you need a purely static version you would have to rewrite the
financial calculations in JavaScript.

For a Python-based interactive dashboard the recommended free hosting options
are:

| Service | URL | Notes |
|---|---|---|
| **Streamlit Community Cloud** | [share.streamlit.io](https://share.streamlit.io) | Recommended — zero config |
| **HuggingFace Spaces** | [huggingface.co/spaces](https://huggingface.co/spaces) | Also free, pick "Streamlit" SDK |
| **Render** | [render.com](https://render.com) | Free tier available |

All three connect directly to a GitHub repository.

---

## Project Structure

```
dashboard/
├── app.py              ← Streamlit dashboard (main entry point)
├── requirements.txt    ← Python dependencies for deployment
└── README.md           ← This file
```
