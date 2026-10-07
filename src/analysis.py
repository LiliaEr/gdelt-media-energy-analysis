from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr


TOPICS = (
    "fossil_energy",
    "nuclear_power",
    "green_energy",
    "climate_policy",
    "sanctions_energy",
    "conflict_energy",
)


@dataclass(frozen=True)
class DataQualitySummary:
    rows: int
    columns: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    missing_days: int
    duplicate_dates: int
    missing_values: int


def load_daily_data(path: str | Path) -> pd.DataFrame:
    """Load and validate the daily topic-level dataset."""
    frame = pd.read_csv(path)
    required = {"date"} | {
        f"{topic}_{suffix}" for topic in TOPICS for suffix in ("count", "tone")
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if frame.empty:
        raise ValueError("The daily dataset is empty.")
    frame["date"] = pd.to_datetime(frame["date"], format="%d/%m/%Y", errors="raise")
    if frame["date"].isna().any():
        raise ValueError("The daily dataset contains missing dates.")
    numeric = frame[[f"{t}_{suffix}" for t in TOPICS for suffix in ("count", "tone")]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Missing or non-finite numeric values")
    counts = frame[[f"{t}_count" for t in TOPICS]]
    if (counts % 1 != 0).any().any():
        raise ValueError("Counts must be integers")
    if (frame[[f"{t}_tone" for t in TOPICS]].abs() > 100).any().any():
        raise ValueError("Tone must lie in [-100, 100]")
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError("The daily dataset contains duplicate dates.")
    if (frame[[f"{t}_count" for t in TOPICS]] < 0).any().any():
        raise ValueError("Mention counts must be non-negative.")
    expected = pd.date_range(frame["date"].min(), frame["date"].max(), freq="D")
    if len(expected.difference(frame["date"])):
        raise ValueError("The daily dataset contains missing days.")
    return frame


def summarize_quality(frame: pd.DataFrame) -> DataQualitySummary:
    if frame.empty:
        return DataQualitySummary(
            rows=0,
            columns=len(frame.columns),
            start_date=pd.NaT,
            end_date=pd.NaT,
            missing_days=0,
            duplicate_dates=0,
            missing_values=0,
        )
    expected = pd.date_range(frame["date"].min(), frame["date"].max(), freq="D")
    return DataQualitySummary(
        rows=len(frame),
        columns=len(frame.columns),
        start_date=frame["date"].min(),
        end_date=frame["date"].max(),
        missing_days=len(expected.difference(frame["date"])),
        duplicate_dates=int(frame["date"].duplicated().sum()),
        missing_values=int(frame.isna().sum().sum()),
    )


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Adjust finite p-values; preserve missing results and index."""
    values = p_values.astype(float)
    valid = values.notna() & np.isfinite(values)
    if ((values[valid] < 0) | (values[valid] > 1)).any():
        raise ValueError("p-values must lie in [0, 1]")
    out = pd.Series(np.nan, index=values.index, dtype=float)
    v = values[valid].to_numpy()
    order = np.argsort(v)
    if len(v):
        adj = v[order] * len(v) / np.arange(1, len(v) + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1].clip(0, 1)
        restored = np.empty_like(adj)
        restored[order] = adj
        out.loc[valid] = restored
    return out


def differenced_topic_correlations(frame: pd.DataFrame) -> pd.DataFrame:

    rows = []
    for topic in TOPICS:
        pair = frame[[f"{topic}_count", f"{topic}_tone"]].diff().dropna()
        ranks = pair.rank()
        rho = float(spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic)
        x = sm.add_constant(ranks.iloc[:, 0].to_numpy())
        row = {"topic": topic, "spearman_rho": rho}
        for lag in (14, 28):
            fit = sm.OLS(ranks.iloc[:, 1].to_numpy(), x).fit(
                cov_type="HAC", cov_kwds={"maxlags": lag})
            row[f"p_hac_{lag}"] = float(fit.pvalues[1])
        rows.append(row)
    result = pd.DataFrame(rows)
    for lag in (14, 28):
        result[f"q_hac_{lag}"] = benjamini_hochberg(result[f"p_hac_{lag}"])
    return result.sort_values("spearman_rho", key=abs, ascending=False).reset_index(drop=True)


def _validate_event_design(sample: pd.DataFrame, design: pd.DataFrame) -> None:
    if sample.empty or not (sample["post"] == 0).any() or not (sample["post"] == 1).any():
        raise ValueError("Event window must contain observations before and after the event.")
    if not np.isfinite(sample.iloc[:, 1].to_numpy(dtype=float)).all():
        raise ValueError("Event outcome must contain only finite values.")
    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise ValueError("Event design is rank deficient.")
    if len(sample) <= design.shape[1]:
        raise ValueError("Event window has insufficient residual degrees of freedom.")


def event_study_hac(
    frame: pd.DataFrame,
    outcome: str,
    event_date: str | pd.Timestamp,
    window_days: int = 60,
    hac_lags: int = 14,
) -> dict[str, float | int | str]:

    event = pd.Timestamp(event_date)
    sample = frame.loc[
        frame["date"].between(event - pd.Timedelta(window_days, "D"),
                              event + pd.Timedelta(window_days, "D")),
        ["date", outcome],
    ].copy()
    sample["t"] = (sample["date"] - event).dt.days.astype(float)
    sample["post"] = (sample["date"] >= event).astype(float)
    sample["post_trend"] = sample["post"] * sample["t"]
    weekday = pd.get_dummies(sample["date"].dt.dayofweek, prefix="dow", drop_first=True, dtype=float)
    design = pd.concat([sample[["t", "post", "post_trend"]], weekday], axis=1)
    design = sm.add_constant(design, has_constant="add").astype(float)
    _validate_event_design(sample, design)
    if (sample[outcome] < 0).any():
        raise ValueError("Event counts must be non-negative.")
    model = sm.OLS(np.log1p(sample[outcome].astype(float)), design).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags}
    )
    beta = float(model.params["post"])
    return {
        "outcome": outcome,
        "event_date": event.date().isoformat(),
        "n_days": int(len(sample)),
        "level_change_pct": float(np.expm1(beta) * 100),
        "ci_low_pct": float(np.expm1(beta - 1.96 * model.bse["post"]) * 100),
        "ci_high_pct": float(np.expm1(beta + 1.96 * model.bse["post"]) * 100),
        "p_value_hac": float(model.pvalues["post"]),
        "post_slope": float(model.params["post_trend"]),
        "r_squared": float(model.rsquared),
    }


def event_study_level_hac(
    frame: pd.DataFrame,
    outcome: str,
    event_date: str | pd.Timestamp,
    window_days: int = 60,
    hac_lags: int = 14,
) -> dict[str, float | int | str]:
    """Local level/slope intervention for continuous outcomes such as AvgTone."""
    event = pd.Timestamp(event_date)
    sample = frame.loc[
        frame["date"].between(event - pd.Timedelta(window_days, "D"),
                              event + pd.Timedelta(window_days, "D")),
        ["date", outcome],
    ].dropna().copy()
    sample["t"] = (sample["date"] - event).dt.days.astype(float)
    sample["post"] = (sample["date"] >= event).astype(float)
    sample["post_trend"] = sample["post"] * sample["t"]
    weekday = pd.get_dummies(sample["date"].dt.dayofweek, prefix="dow", drop_first=True, dtype=float)
    design = pd.concat([sample[["t", "post", "post_trend"]], weekday], axis=1)
    design = sm.add_constant(design, has_constant="add").astype(float)
    _validate_event_design(sample, design)
    model = sm.OLS(sample[outcome].astype(float), design).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags}
    )
    beta = float(model.params["post"])
    margin = 1.96 * float(model.bse["post"])
    return {
        "outcome": outcome,
        "event_date": event.date().isoformat(),
        "n_days": int(len(sample)),
        "level_change": beta,
        "ci_low": beta - margin,
        "ci_high": beta + margin,
        "p_value_hac": float(model.pvalues["post"]),
        "post_slope": float(model.params["post_trend"]),
        "r_squared": float(model.rsquared),
    }
