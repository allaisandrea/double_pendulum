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
    mse, _, _ = rollout_losses(model, obs, present, action, 3)
    mse.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_gaussian_nll_is_the_normal_log_likelihood_and_beta_weighs_it_by_the_variance():
    import math

    from world_model.model_lib import gaussian_nll

    error, logvar = torch.tensor([0.5, -1.0]), torch.tensor([0.0, math.log(4.0)])
    expected = -torch.distributions.Normal(0.0, logvar.mul(0.5).exp()).log_prob(error)
    assert torch.allclose(gaussian_nll(error, logvar), expected)
    assert torch.allclose(gaussian_nll(error, logvar, 0.5), expected * torch.tensor([1.0, 2.0]))


def test_a_stochastic_model_predicts_its_mean_at_tau_0_and_samples_its_spread():
    from world_model.model_lib import WorldModel, load_world_model, losses

    torch.manual_seed(0)
    model = WorldModel(4, 8, 1, 0.1, stochastic=True)
    obs = torch.nn.functional.normalize(torch.randn(1, 4, 3, 2), dim=-1).expand(4096, -1, -1, -1)
    present = torch.ones(4096, 4, 3, dtype=torch.bool)
    action = torch.zeros(4096, 4)
    delta, _, logvar = model(obs, present, action)
    mean, *_ = model.predict(obs, present, action)
    ref = obs[:, -1]
    assert torch.allclose(mean, ref + delta * 0.1)
    drawn, *_ = model.predict(obs, present, action, tau=0.5, generator=torch.Generator().manual_seed(0))
    sd = ((drawn - mean) / 0.1).std(0)
    assert torch.allclose(sd, 0.5 * logvar[0].mul(0.5).exp(), rtol=0.05)
    again, *_ = model.predict(obs, present, action, tau=0.5, generator=torch.Generator().manual_seed(0))
    assert torch.equal(drawn, again)
    fit, bce, mse = losses(model, obs, present, action, obs[:, -1], present[:, -1], beta=0.5)
    fit.backward()
    assert model.head.weight.grad is not None


def test_a_deterministic_model_ignores_tau():
    from world_model.model_lib import WorldModel

    model = WorldModel(4, 8, 1, 0.1)
    obs = torch.nn.functional.normalize(torch.randn(2, 4, 3, 2), dim=-1)
    present, action = torch.ones(2, 4, 3, dtype=torch.bool), torch.zeros(2, 4)
    assert torch.equal(model.predict(obs, present, action)[0], model.predict(obs, present, action, tau=1.0)[0])
