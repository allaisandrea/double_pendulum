import pytest
import torch

from common.data_lib import Batch
from world_model.evaluation_lib import one_minus_r2, rollout
from world_model.model_lib import WorldModel


def test_copy_last_scores_about_one_and_a_perfect_prediction_zero():
    change = torch.randn(1000, 3, 2)
    mask = torch.ones(1000, 3, dtype=torch.bool)
    assert one_minus_r2(torch.zeros_like(change), change, mask) == pytest.approx(1.0, abs=0.05)
    assert one_minus_r2(change, change, mask) == 0.0


def test_rollout_feeds_predictions_back_with_the_recorded_actions():
    window, horizon = 4, 3
    model = WorldModel(window, 8, 1, 0.1)
    seen_actions = []
    real_forward = model.forward

    def forward(obs, present, action):
        seen_actions.append(action[0].tolist())
        return real_forward(obs, present, action)

    model.forward = forward
    obs = torch.nn.functional.normalize(torch.randn(1, window + horizon, 3, 2), dim=-1)
    present = torch.ones(1, window + horizon, 3, dtype=torch.bool)
    action = torch.arange(window + horizon, dtype=torch.float32)[None]

    preds, _, _ = rollout(model, Batch(obs, present, action), horizon)
    assert preds.shape == (1, horizon, 3, 2)
    assert preds.norm(dim=-1).allclose(torch.ones(1, horizon, 3))
    assert seen_actions == [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]
