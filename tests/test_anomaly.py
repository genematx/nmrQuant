import numpy as np
import pandas as pd
import pytest
from prophet import Prophet

from uv_monitor.anomaly import DailyAnomalyDetector


@pytest.fixture
def sample_data():
    index = pd.date_range("2022-01-01", "2022-12-31", freq="D")
    data = pd.Series(np.random.randn(len(index)), index=index)
    return data


def test_prepare_data_for_prophet(sample_data):
    detector = DailyAnomalyDetector()
    prepared_data = detector._DailyAnomalyDetector__prepare_data_for_prophet(
        sample_data
    )
    assert set(prepared_data.columns) == set(["ds", "y"])
    assert isinstance(prepared_data, pd.DataFrame)


def test_fit_prophet(sample_data):
    detector = DailyAnomalyDetector()
    detector.fit(sample_data)
    assert isinstance(detector._model, Prophet)


def test_predict(sample_data):
    detector = DailyAnomalyDetector()
    detector.fit(sample_data)
    preds = detector.predict(sample_data)
    assert set(preds.columns) == set(
        ["y", "yhat", "yhat_lower", "yhat_upper", "is_anomaly"]
    )
    assert isinstance(preds, pd.DataFrame)
    assert preds.shape[0] == sample_data.shape[0]


def test_clip_negative(sample_data):
    detector = DailyAnomalyDetector(clip_negative=True)
    detector.fit(sample_data)
    preds_clip = detector.predict(sample_data)

    detector_no_clip = DailyAnomalyDetector(clip_negative=False)
    detector_no_clip.fit(sample_data)
    preds_no_clip = detector_no_clip.predict(sample_data)

    assert all(preds_clip["yhat"] >= 0)
    assert all(preds_clip["yhat_lower"] >= 0)
    assert all(preds_clip["yhat_upper"] >= 0)

    indx_no_clip = (
        (preds_no_clip["yhat"] > 0)
        & (preds_no_clip["yhat_lower"] > 0)
        & (preds_no_clip["yhat_upper"] > 0)
    )
    assert all(
        preds_no_clip.loc[indx_no_clip, "is_anomaly"]
        == preds_clip.loc[indx_no_clip, "is_anomaly"]
    )
