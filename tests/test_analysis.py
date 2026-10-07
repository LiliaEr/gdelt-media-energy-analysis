from pathlib import Path

import pandas as pd
import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from src.analysis import TOPICS, benjamini_hochberg, load_daily_data, summarize_quality


DATA = Path(__file__).parents[1] / "data" / "raw" / "gdelt_energy_daily.csv"


def test_daily_dataset_is_complete_and_unique():
    frame = load_daily_data(DATA)
    quality = summarize_quality(frame)
    assert quality.rows == 1096
    assert quality.start_date == pd.Timestamp("2022-01-01")
    assert quality.end_date == pd.Timestamp("2024-12-31")
    assert quality.missing_days == 0
    assert quality.duplicate_dates == 0
    assert quality.missing_values == 0


def test_expected_topic_columns_are_present():
    frame = load_daily_data(DATA)
    for topic in TOPICS:
        assert f"{topic}_count" in frame
        assert f"{topic}_tone" in frame


def test_bh_adjustment_is_monotone_in_rank():
    raw = pd.Series([0.01, 0.04, 0.03, 0.20])
    adjusted = benjamini_hochberg(raw)
    ranked = pd.DataFrame({"raw": raw, "adjusted": adjusted}).sort_values("raw")
    assert ranked["adjusted"].is_monotonic_increasing
    assert adjusted.between(0, 1).all()


def test_bh_matches_reference_and_preserves_missing_tests():
    raw = pd.Series([0.04, np.nan, 0.001, 0.8, 0.015], index=list('abcde'))
    actual = benjamini_hochberg(raw)
    expected = multipletests(raw.dropna(), method='fdr_bh')[1]
    np.testing.assert_allclose(actual.dropna(), expected)
    assert pd.isna(actual['b'])


@pytest.mark.parametrize('column,value', [
    ('green_energy_tone', 101), ('fossil_energy_count', 0.5),
    ('nuclear_power_count', np.inf), ('climate_policy_count', -1),
])
def test_loader_rejects_invalid_measurements(tmp_path, column, value):
    data = pd.read_csv(DATA).astype({column:float})
    data.loc[0,column] = value
    path = tmp_path / 'invalid.csv'
    data.to_csv(path,index=False)
    with pytest.raises(ValueError):
        load_daily_data(path)


def test_intervention_recovers_known_synthetic_step():
    from src.analysis import event_study_hac
    rng = np.random.default_rng(42)
    dates = pd.date_range('2022-01-01',periods=181)
    event = dates[90]
    relative = np.arange(181)-90
    log_count = 7 + .001*relative + np.log(1.8)*(relative>=0)
    log_count += .05*np.sin(2*np.pi*np.arange(181)/7) + rng.normal(0,.01,181)
    frame = pd.DataFrame({'date':dates,'count':np.expm1(log_count)})
    fit = event_study_hac(frame,'count',event,window_days=60)
    assert abs(fit['level_change_pct']-80) < 3
    assert fit['ci_low_pct'] < 80 < fit['ci_high_pct']


@pytest.mark.parametrize("case", ["gap", "missing_date", "empty"])
def test_loader_rejects_invalid_calendar(tmp_path, case):
    data = pd.read_csv(DATA)
    if case == "gap":
        data = data.drop(index=10)
    elif case == "missing_date":
        data.loc[0, "date"] = np.nan
    else:
        data = data.iloc[:0]
    path = tmp_path / "invalid_calendar.csv"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_daily_data(path)


@pytest.mark.parametrize("function_name", ["event_study_hac", "event_study_level_hac"])
@pytest.mark.parametrize("event", ["2021-12-31", "2022-01-01", "2022-04-01", "2023-01-01"])
def test_intervention_rejects_one_sided_or_empty_window(function_name, event):
    from src import analysis
    frame = pd.DataFrame({"date": pd.date_range("2022-01-01", periods=90), "value": 10.0})
    with pytest.raises(ValueError, match="before and after"):
        getattr(analysis, function_name)(frame, "value", event)


@pytest.mark.parametrize("function_name", ["event_study_hac", "event_study_level_hac"])
def test_intervention_rejects_rank_deficient_window(function_name):
    from src import analysis
    frame = pd.DataFrame({"date": pd.date_range("2022-01-01", periods=5), "value": 10.0})
    with pytest.raises(ValueError, match="rank deficient"):
        getattr(analysis, function_name)(frame, "value", "2022-01-03", window_days=2)


def test_level_intervention_recovers_known_step():
    from src.analysis import event_study_level_hac
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", periods=181)
    relative = np.arange(181) - 90
    values = .001 * relative - .8 * (relative >= 0) + rng.normal(0, .01, 181)
    frame = pd.DataFrame({"date": dates, "tone": values})
    fit = event_study_level_hac(frame, "tone", dates[90])
    assert abs(fit["level_change"] + .8) < .03


def test_summarize_quality_empty_frame():
    empty = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")})
    summary = summarize_quality(empty)
    assert summary.rows == 0
    assert summary.missing_days == 0
    assert summary.duplicate_dates == 0
    assert pd.isna(summary.start_date)
    assert pd.isna(summary.end_date)


def test_differenced_topic_correlations():
    from src.analysis import differenced_topic_correlations
    frame = load_daily_data(DATA)
    res = differenced_topic_correlations(frame)
    assert len(res) == len(TOPICS)
    assert set(res["topic"]) == set(TOPICS)
    for col in ("spearman_rho", "p_hac_14", "p_hac_28", "q_hac_14", "q_hac_28"):
        assert col in res.columns
        assert res[col].notna().all()


def test_advanced_models_distributed_lag_and_cop():
    from src.advanced_models import multi_event_distributed_lag, cop_event_time_model
    frame = load_daily_data(DATA)
    events = {"2022-02-24": "Invasion", "2022-09-26": "NordStream"}
    coefs, joint = multi_event_distributed_lag(frame, "fossil_energy_count", events)
    assert len(coefs) == len(events) * 3
    assert len(joint) == len(events)
    assert set(joint["event"]) == {"Invasion", "NordStream"}

    with pytest.raises(ValueError, match="At least one event"):
        multi_event_distributed_lag(frame, "fossil_energy_count", {})

    cop_dates = {"2022-11-06": "COP27", "2023-11-30": "COP28"}
    cop_res = cop_event_time_model(frame, "climate_policy_count", cop_dates)
    assert len(cop_res) == 5
    assert (cop_res["p_value_fdr"] >= 0).all() and (cop_res["p_value_fdr"] <= 1).all()

    with pytest.raises(ValueError, match="At least one COP event"):
        cop_event_time_model(frame, "climate_policy_count", {})


def test_var_changes_by_topic():
    from src.advanced_models import var_changes_by_topic
    frame = load_daily_data(DATA)
    var_res = var_changes_by_topic(frame)
    assert len(var_res) == len(TOPICS)
    assert set(var_res["topic"]) == set(TOPICS)
    assert "status" in var_res.columns
    assert "tone_to_count_q" in var_res.columns

