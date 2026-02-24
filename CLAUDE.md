# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A two-stage data pipeline + Streamlit dashboard for a travel agency's sales data:

1. **`main.py`** — ETL script: reads annual Excel files (stored on Windows/OneDrive), normalizes them, and outputs `ventas_viajes.csv`.
2. **`app.py`** — Streamlit dashboard: reads `ventas_viajes.csv` and displays commercial/financial KPIs and interactive Plotly charts.

## Commands

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync

# Generate/refresh the CSV from Excel sources (run first)
uv run python main.py

# Launch the Streamlit dashboard
uv run streamlit run app.py
```

## Architecture

### Data flow
```
Excel files (Windows/OneDrive) → main.py → ventas_viajes.csv → app.py → Streamlit UI
```

### main.py — ETL pipeline

- `ARCHIVOS_POR_ANIO` maps year → Windows path (e.g. `C:\Users\...`). `ruta_windows_a_wsl()` converts these to WSL `/mnt/c/...` paths automatically.
- Header detection is fuzzy: `_detectar_fila_cabecera()` searches the first 40 rows for the real header row (Excel files contain title rows above the data).
- `ALIAS_EXCEL` maps variant/misspelled column names to canonical Excel names. `MAPEO_EXCEL_A_APP` then maps those to the names that `app.py` expects.
- `COLUMNAS_APP` is the contract: every column in that list must exist in the output CSV. Missing columns are filled with `pd.NA`.

### app.py — Streamlit dashboard

- Loads CSV with `@st.cache_data`. Derives computed columns on load: `Comisión Total`, `Margen Bruto %`, `Lead Time (Días)`, and `Estado de Pago`.
- Sidebar filters by date range, airline, and route — all charts and KPIs react to the filtered `df_filtrado`.
- KPIs: total sales, total commission, avg gross margin %, avg ticket, avg lead time.
- Charts: monthly sales vs commissions (bar+line dual-axis), top-10 routes by profitability, airline market share treemap, price distribution histogram, lead time vs price scatter.
- Operational table at the bottom lists pending payments (`Estado de Pago == 'Pendiente'`).

## Adding a New Year

Add an entry to `ARCHIVOS_POR_ANIO` in `main.py` and re-run `python main.py`.

## Column Name Contract

If the Excel source changes column names, update `ALIAS_EXCEL` in `main.py`. If `app.py` needs new columns, add them to `COLUMNAS_APP` and handle them in `convertir_columnas_app()`.
