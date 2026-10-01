import math

import torch

from world_model.flow_lib import FlowWorldModel, flow_losses, rotate, yaw_change


def unit(*shape):
    return torch.nn.functional.normalize(torch.randn(*shape, 2), dim=-1)


def test_rotating_by_the_yaw_change_gives_the_observation():
    ref, obs = unit(100), unit(100)
    angle = yaw_change(ref, obs)
    assert (angle.abs() <= math.pi).all()
    assert torch.allclose(rotate(ref, angle), obs, atol=1e-6)


def gaussian_flow(model: FlowWorldModel, mu: torch.Tensor, sigma: torch.Tensor):
    """Makes the model's velocity the straight path's exact field from
    N(0, I) to N(mu, sigma²), per coordinate."""
    def velocity(cond, x, t):
        t = t[:, None]
        var = (1 - t).pow(2) + t.pow(2) * sigma.pow(2)
        return mu + (x - t * mu) * (t * sigma.pow(2) - (1 - t)) / var

    model.velocity = velocity


def test_the_flow_samples_and_scores_the_distribution_its_velocity_carries_noise_to():
    model = FlowWorldModel(4, 8, 1, [0.1, 0.2, 0.5], 8, 2, flow_steps=32)
    mu, sigma = torch.tensor([0.5, -1.0, 2.0]), torch.tensor([0.3, 1.5, 0.8])
    gaussian_flow(model, mu, sigma)
    x0 = torch.randn(64, 3)
    assert torch.allclose(model.integrate(None, x0), mu + sigma * x0, atol=1e-3)

    obs, present, action = unit(64, 4, 3), torch.ones(64, 4, 3, dtype=torch.bool), torch.zeros(64, 4)
    change = (mu + sigma * torch.randn(64, 3)) * model.delta_scale
    # The density of the change in radians: the normalised one, less the scale.
    expected = torch.distributions.Normal(mu, sigma).log_prob(change / model.delta_scale).sum(-1)
    expected -= model.delta_scale.log().sum()
    assert torch.allclose(model.log_prob(obs, present, action, change), expected, atol=1e-3)


def test_predict_turns_the_reference_by_the_drawn_change():
    torch.manual_seed(0)
    model = FlowWorldModel(4, 8, 1, [0.1, 0.1, 0.1], 8, 2)
    obs, present, action = unit(5, 4, 3), torch.ones(5, 4, 3, dtype=torch.bool), torch.zeros(5, 4)
    nxt, logit, ref, has_ref = model.predict(obs, present, action)
    assert nxt.norm(dim=-1).allclose(torch.ones(5, 3)) and logit.shape == (5, 3) and has_ref.all()
    assert torch.equal(nxt, model.predict(obs, present, action)[0])
    draw = lambda: model.predict(obs, present, action, 1.0, torch.Generator().manual_seed(1))[0]
    assert torch.equal(draw(), draw()) and not torch.allclose(draw(), nxt)


def test_the_loss_ignores_tags_unseen_in_the_next_frame_and_trains_every_weight():
    torch.manual_seed(0)
    model = FlowWorldModel(4, 8, 1, [0.1, 0.1, 0.1], 8, 2)
    obs, present, action = unit(16, 4, 3), torch.ones(16, 4, 3, dtype=torch.bool), torch.zeros(16, 4)
    target, target_present = unit(16, 3), torch.rand(16, 3) > 0.3
    other = torch.where(target_present[..., None], target, unit(16, 3))
    losses = [flow_losses(model, obs, present, action, t, target_present, torch.Generator().manual_seed(0))
              for t in (target, other)]
    assert torch.equal(losses[0][0], losses[1][0])
    fm, bce = losses[0]
    (fm + bce).backward()
    assert all(p.grad is not None for p in model.parameters())
