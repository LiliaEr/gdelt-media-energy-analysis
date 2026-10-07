# Energy in the News — GDELT, 2022–2024

**Lilia Erofeevskaya · Python · time-series analysis**

This study examines changes in the frequency and tone of six energy-related topics: oil and gas, nuclear power, green energy, climate policy, sanctions, and conflict. It considers overall trends and changes around major events.

## Explore the analysis

**[Main notebook with charts and findings](notebooks/gdelt_energy_media_analysis.ipynb)**

![Correlations in energy topic frequency](reports/figures/04_count_heatmap.png)

## Main findings

- Green energy and climate policy are strongly correlated in frequency: Pearson’s r = **0.89**. Their tone correlation is weaker: **0.30**.
- Around 24 February 2022, the local model identifies increases in the conflict and sanctions counts. The change in sanctions tone is not statistically significant.
- Around COP summits, the largest positive coefficient corresponds to the first week: **+18.7%**, controlling for lagged values, trend, and seasonality.
- H4, concerning the predictive relationship between tone and frequency, remains inconclusive: the daily VAR models retain residual dependence despite small p-values.

These are observational associations in an existing data export. They do not establish causal effects or measure public opinion.

## Data and SQL

- [Daily indicators](data/raw/gdelt_energy_daily.csv): 1,096 days, 1 January 2022–31 December 2024; 13 columns.
- [Geographic aggregation](data/raw/country_article_counts.csv): location-entry counts, not necessarily unique article counts.
- [SQL script](sql/gdelt_energy_unified.sql).

Topics overlap, and total daily news volume is unavailable. The sanctions and conflict topic definitions require validation against source texts. The notebook uses local exports and does not require BigQuery access.

## Running the project

Python 3.11 or later. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/run_notebook.py
```

On macOS or Linux, create the environment with `python3 -m venv .venv` and activate it with `source .venv/bin/activate`, then run the installation, test, and notebook commands above. In VS Code, select the kernel from this environment.

The notebook is edited directly: `notebooks/gdelt_energy_media_analysis.ipynb` is the sole source of its text and structure. `run_notebook.py` executes every cell and saves the outputs. Figures and calculated tables are written to `reports/figures` and `reports/tables`; the raw CSV files are unchanged.

## Project structure

| Path | Contents |
|---|---|
| `notebooks/` | Main analysis with saved outputs |
| `data/raw/` | Two input CSV files |
| `sql/` | SQL queries |
| `src/` | Validation, correlations, and H1–H4 models |
| `scripts/` | Notebook execution |
| `tests/` | Data and calculation checks |
| `reports/` | Event date sources, selected figures, and calculated tables |
| `archive/` | Local archives; excluded from publication on GitHub |

## Validation and limitations

[Event date sources](reports/event_sources.md)
