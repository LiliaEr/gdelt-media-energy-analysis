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
