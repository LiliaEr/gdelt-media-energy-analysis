from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

from .analysis import TOPICS, benjamini_hochberg


def _calendar_controls(dates: pd.Series, trend: np.ndarray) -> pd.DataFrame:
    controls = pd.DataFrame(index=dates.index)
    controls["trend"] = trend / max(len(trend), 1)
    for harmonic in (1, 2):
        controls[f"annual_sin_{harmonic}"] = np.sin(2 * np.pi * harmonic * trend / 365.25)
        controls[f"annual_cos_{harmonic}"] = np.cos(2 * np.pi * harmonic * trend / 365.25)
    weekday = pd.get_dummies(dates.dt.dayofweek, prefix="dow", drop_first=True, dtype=float)
    return pd.concat([controls, weekday], axis=1)


def multi_event_distributed_lag(
    frame: pd.DataFrame,
    outcome: str,
    events: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Joint dynamic regression with 0–7, 8–14 and 15–30 day event lag bins."""
    work = frame[["date", outcome]].copy()
    work["y"] = np.log1p(work[outcome].astype(float))
    work["lag1"] = work["y"].shift(1)
    work["lag7"] = work["y"].shift(7)
    trend = np.arange(len(work), dtype=float)
    design = _calendar_controls(work["date"], trend)
    design[["lag1", "lag7"]] = work[["lag1", "lag7"]]

    event_columns: dict[str, list[str]] = {}
    bins = ((0, 7), (8, 14), (15, 30))
    for number, (date, name) in enumerate(events.items(), start=1):
        relative = (work["date"] - pd.Timestamp(date)).dt.days
        event_columns[name] = []
        for start, end in bins:
            column = f"event_{number}_lag_{start}_{end}"
            design[column] = relative.between(start, end).astype(float)
            event_columns[name].append(column)

    valid = design.notna().all(axis=1)
    x = sm.add_constant(design.loc[valid], has_constant="add").astype(float)
    if np.linalg.matrix_rank(x.to_numpy()) != x.shape[1]:
        raise ValueError("Event design is rank deficient")
    fitted = sm.OLS(work.loc[valid, "y"], x).fit(cov_type="HAC", cov_kwds={"maxlags": 14})

    coefficient_rows = []
    joint_rows = []
    for name, columns in event_columns.items():
        for column in columns:
            beta = float(fitted.params[column])
            se = float(fitted.bse[column])
            coefficient_rows.append({
                "event": name,
                "lag_bin": column.rsplit("_lag_", 1)[1].replace("_", "–"),
                "effect_pct": float(np.expm1(beta) * 100),
                "ci_low_pct": float(np.expm1(beta - 1.96 * se) * 100),
                "ci_high_pct": float(np.expm1(beta + 1.96 * se) * 100),
                "p_value": float(fitted.pvalues[column]),
            })
        restriction = np.zeros((len(columns), len(fitted.params)))
        for row, column in enumerate(columns):
            restriction[row, list(fitted.params.index).index(column)] = 1
        test = fitted.wald_test(restriction, scalar=True)
        joint_rows.append({"event": name, "joint_p_value": float(test.pvalue)})

    coefficients = pd.DataFrame(coefficient_rows)
    coefficients["p_value_fdr"] = benjamini_hochberg(coefficients["p_value"])
    joint = pd.DataFrame(joint_rows)
    joint["joint_p_fdr"] = benjamini_hochberg(joint["joint_p_value"])
    return coefficients, joint


def cop_event_time_model(
    frame: pd.DataFrame,
    outcome: str,
    cop_dates: dict[str, str],
) -> pd.DataFrame:
    """Full-series dynamic regression with pooled COP lead/lag indicators."""
    work = frame[["date", outcome]].copy()
    work["y"] = np.log1p(work[outcome].astype(float))
    work["lag1"] = work["y"].shift(1)
    work["lag7"] = work["y"].shift(7)
    trend = np.arange(len(work), dtype=float)
    design = _calendar_controls(work["date"], trend)
    design[["lag1", "lag7"]] = work[["lag1", "lag7"]]

    bins = {
        "lead_30_15": (-30, -15),
        "lead_14_1": (-14, -1),
        "event_0_6": (0, 6),
        "lag_7_14": (7, 14),
        "lag_15_30": (15, 30),
    }
    for label, (start, end) in bins.items():
        indicator = pd.Series(0.0, index=work.index)
        for date in cop_dates:
            relative = (work["date"] - pd.Timestamp(date)).dt.days
            indicator = np.maximum(indicator, relative.between(start, end).astype(float))
        design[label] = indicator

    valid = design.notna().all(axis=1)
    x = sm.add_constant(design.loc[valid], has_constant="add").astype(float)
    if np.linalg.matrix_rank(x.to_numpy()) != x.shape[1]:
        raise ValueError("Event design is rank deficient")
    fitted = sm.OLS(work.loc[valid, "y"], x).fit(cov_type="HAC", cov_kwds={"maxlags": 14})
    rows = []
    for label in bins:
        beta = float(fitted.params[label])
        se = float(fitted.bse[label])
        rows.append({
            "event_time": label,
            "effect_pct": float(np.expm1(beta) * 100),
            "ci_low_pct": float(np.expm1(beta - 1.96 * se) * 100),
            "ci_high_pct": float(np.expm1(beta + 1.96 * se) * 100),
            "p_value": float(fitted.pvalues[label]),
        })
    output = pd.DataFrame(rows)
    output["p_value_fdr"] = benjamini_hochberg(output["p_value"])
    return output


def var_changes_by_topic(frame: pd.DataFrame, max_lags: int = 14) -> pd.DataFrame:
    """Exploratory tone-to-count Granger tests on stationary first differences.

    All topics use the same transformation; no unsupported I(1) classification.
    AIC=0 is retained as no lagged model, rather than silently forced to one.
    Residual whiteness and stability qualify, rather than hide, model failures.
    """
    rows = []
    for topic in TOPICS:
        changes = pd.DataFrame({
            "count": np.log1p(frame[f"{topic}_count"].astype(float)).diff(),
            "tone": frame[f"{topic}_tone"].astype(float).diff(),
        }).dropna().reset_index(drop=True)
        p_count = float(adfuller(changes["count"], autolag="AIC")[1])
        p_tone = float(adfuller(changes["tone"], autolag="AIC")[1])
        row = {"topic": topic, "adf_p_count_change": p_count,
               "adf_p_tone_change": p_tone, "lag": 0,
               "tone_to_count_p": np.nan, "stable": False,
               "whiteness_p": np.nan}
        if p_count >= .05 or p_tone >= .05:
            row["status"] = "stationarity not established"
        else:
            fit = VAR(changes).fit(maxlags=max_lags, ic="aic", trend="c")
            row["lag"] = int(fit.k_ar)
            if fit.k_ar == 0:
                row["status"] = "AIC selected no lags"
            else:
                row["tone_to_count_p"] = float(fit.test_causality("count", ["tone"], kind="f").pvalue)
                row["stable"] = bool(fit.is_stable())
                row["whiteness_p"] = float(fit.test_whiteness(nlags=max(28, fit.k_ar+7), adjusted=True).pvalue)
                row["status"] = "diagnostics passed" if row["stable"] and row["whiteness_p"] >= .05 else "residual dependence or instability"
        rows.append(row)
    out = pd.DataFrame(rows)
    out["tone_to_count_q"] = benjamini_hochberg(out["tone_to_count_p"])
    return out
