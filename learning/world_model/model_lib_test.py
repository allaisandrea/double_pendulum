import torch

from world_model.model_lib import last_seen


def test_last_seen_takes_each_tags_latest_observation():
    obs = torch.arange(1, 4, dtype=torch.float32)[None, :, None, None].expand(1, 3, 3, 2)
    present = torch.tensor([[[True, True, False], [True, False, False], [False, False, False]]])
    ref, seen = last_seen(obs, present)
    assert ref[0, :, 0].tolist() == [2, 1, 0]
    assert seen[0].tolist() == [True, True, False]


def test_a_one_step_rollout_loss_is_the_one_step_loss():
    from world_model.model_lib import WorldModel, losses, rollout_losses

    torch.manual_seed(0)
    model = WorldModel(4, 8, 1, 0.1)
    obs = torch.nn.functional.normalize(torch.randn(6, 7, 3, 2), dim=-1)
    present = torch.rand(6, 7, 3) > 0.2
    action = torch.randn(6, 7)
    one = losses(model, obs[:, :4], present[:, :4], action[:, :4], obs[:, 4], present[:, 4])
    rolled = rollout_losses(model, obs[:, :5], present[:, :5], action[:, :5], 1)
    assert torch.allclose(one[0], rolled[0]) and torch.allclose(one[1], rolled[1])
    # Three steps average three losses, each through the model's own predictions.
    mse, _ = rollout_losses(model, obs, present, action, 3)
    mse.backward()
    assert all(p.grad is not None for p in model.parameters())
