# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pytest
import yaml

import pioreactor_calibration_shrinkage as shrinkage_module


def test_polynomial_features_builds_design_matrix():
    features = shrinkage_module.polynomial_features([1.0, 2.0], 3)
    expected = np.array([[1.0, 1.0, 1.0], [4.0, 2.0, 1.0]])
    assert features.shape == expected.shape
    assert np.allclose(features, expected)


def test_objective_function_matches_expected_value():
    w = np.array([1.0, 2.0])
    A = np.array([1.0])
    X = np.eye(2)
    y = np.array([1.0, 2.0])
    value = shrinkage_module.objective_function(w, A, [X], [y], lambda_w=0.5, lambda_a=0.3)
    assert value == pytest.approx(2.5)


def test_fit_model_recovers_shared_polynomial_and_scalars():
    np.random.seed(0)
    degree = 2
    true_w = np.array([2.0, 1.0])
    x_values = np.array([0.0, 1.0, 2.0, 3.0])
    design = shrinkage_module.polynomial_features(x_values, degree)
    true_a = np.array([1.0, 1.5])
    X_list = [design, design]
    y_list = [true_a[i] * design.dot(true_w) for i in range(len(true_a))]

    w_est, a_est = shrinkage_module.fit_model(
        X_list,
        y_list,
        d=degree,
        lambda_w=1e-6,
        lambda_a=0.1,
        max_iter=500,
        tol=1e-12,
    )

    assert np.allclose(
        a_est[1] / a_est[0],
        true_a[1] / true_a[0],
        rtol=0.2,
        atol=0.1,
    )
    for expected, predicted in zip(y_list, [a_est[i] * X_list[i].dot(w_est) for i in range(len(X_list))]):
        assert np.allclose(predicted, expected, atol=1e-2)


def test_prepare_data_builds_feature_matrices():
    data = [
        {"recorded_data": {"x": [0.0, 1.0], "y": [1.0, 3.0]}},
        {"recorded_data": {"x": [2.0, 3.0], "y": [5.0, 7.0]}},
    ]
    degree = 2
    X_list, y_list = shrinkage_module.prepare_data(data, degree)

    assert len(X_list) == len(data)
    assert len(y_list) == len(data)
    assert X_list[0].shape == (2, degree)
    assert np.allclose(X_list[0], shrinkage_module.polynomial_features([0.0, 1.0], degree))
    assert np.allclose(y_list[1], np.array([5.0, 7.0]))


def test_simple_prefix_hash_is_stable():
    result = shrinkage_module.simple_prefix_hash("abc")
    assert result == shrinkage_module.simple_prefix_hash("abc")
    assert len(result) == 4


def test_send_calibration_to_worker_sends_yaml(monkeypatch):
    captured = {}

    def fake_resolve(worker: str) -> str:
        captured["worker"] = worker
        return f"http://{worker}"

    def fake_post(url: str, path: str, json: dict):
        captured["url"] = url
        captured["path"] = path
        captured["json"] = json

        class Response:
            def raise_for_status(self):
                return None

        return Response()

    monkeypatch.setattr(shrinkage_module, "resolve_to_address", fake_resolve)
    monkeypatch.setattr(shrinkage_module, "post_into", fake_post)

    calibration = {"calibration_name": "example", "curve_data_": [1.0, 2.0]}

    response = shrinkage_module.send_calibration_to_worker("worker-1", "device-a", calibration)

    assert captured["url"] == "http://worker-1"
    assert captured["path"] == "/unit_api/calibrations/device-a"
    assert yaml.safe_load(captured["json"]["calibration_data"]) == calibration
    assert hasattr(response, "raise_for_status")
